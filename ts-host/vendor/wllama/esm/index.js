var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b5-b35efc8";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true
    });
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
