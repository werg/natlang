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
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
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
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
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
  "compat": "H4sIAAAAAAAAA+S9eY/dRpYvWEB5kcrWLqVSqS21k9qsXKxypWVVu2yX2+1yl9u1NLr7NQJxybj3spKbGOTNTOMhMQ+Y+Wf+HAwwwHyl+SzzFQaDc04EGREM8l5l+dUr4P0j5T3nRDBIBiNOnOV3/p+f/exn/++jn/3scO1nP7sqMhlVSVmLnE2aJK2TnE0rIX4xm2Up45Oiqm+nKc84y4pYpGzCpdjbiyrBa8Fqkcui+rAoRcXrotrMxcH1qMiyImd/kUW+t3ds/ArCB8YvltQiswWQFITrjEWHnPE0LSK4iDiMRFknRX6ZGDC6jnhK1vHe3vPnz9cYi1Kez1jE05TVosqSnNfivUxkUVZeSdOMzSpezllU5LU4rPf2oskZusk4ZklWpu+Iw3L6nqyrVOQfTYoi3TRvPC14LKq9vZmo2b44etXkMpnlIt5Mnj9//voSjW3W8CpmlUgFl8Ki8ehNk1TiHF6xEnLOS8F24kfm/S942gjnmSAtCB8tHc80LXj9+jT2D7d4NkqTkm65u1UpapbzTNCvrEnxxu9h5/gg9av87LNX0ZxXT5EKN3gVWyQS+05mTdFIVhUH8sO2p4zXO2rkpZixJq+TlJW8kjDK4wFOEK5ZE6a9/g2jge5EM5/YT3E7PstYfJTzLIlYxGVts3djGmMpqqypxUV6iGkxY0leiyrn6QfIzydHtZBn1XWBz+P48zwt5hnP8709fB18krDFDtvaZi/29iZcJhEO/FUnBg/LZOEEnFc36LLzklc8k3t7ORPZJGZzwWO2/8gzO497tCC80usE2q97qWx/cYmmdyWY5Jlg+Dgu9YSn03eiQk4v9xhVUV+xXg6Mgye5vO8Zbo80dsOLC9anAYvMo4NZ2ezt/SbJ46+roim/yuvqaG/v2CYE4XV/tws2e8Ov4f3CKhazaVEdQOfisOS5fkT45cBY2x8D3e1Dd7uLIonps2CshlVnwlOeR4LxaS0qluRSVPWrTgBePc21RSIO2Hb8wPxgYxEl+FptUhB6ntRhKaqaNVLEH+gPOuL1OzLJp+eRMC2qjNO3vOl5G/QM0qLiLMsu9j/dyzTjxYFawnEZeD8pqySvp7Q4wLoCH/jZduFI8rKpT8mcpN6riiaPp2vmMkOXzYsqMx7DTnzNs8dkuL7fW7K1sDf7C9qJZMRT8eWKXyO+koyXTzfpazzueEH43kzUIl+8w+siWc84iys2TXnEYNkueMwiHs3Fqal6GHT7dcVzWRZS0M+qKAU+sovW9yHyOAhvDr4OeC7X/pLkf+F7e9MmjxivZpIW8LKQt/ockcumEiwqmrw2Huhu/Jw+lt9P/iKi+jdcildE+KLISnXHHS8IL7pveyt+Jy1m0/emB1VSi0/MeT7ncs5qPknFq45CfTIWF6wSIPGqrhrx+qpnLYG5e+9ATGZlw8qkFGmSw5bmUILw12OvMhY1T+BDFYei6q2tM1EH4VnYnl6/3oyKXNYPw+v06fNoX+QxmzTTKXygks0LWdN8rjKJb+CMrCcJg+m99TIVv1QvMJrzmmVyxmKRJllSm3tWnxeE9xxeXzoIH5kU/HgcKaQF4ctlezsrKsaril4Iryp+BA/ismdH3rrT2zZpPYTeKlEKriYStGHbgxPpN8106plIH6qnXFWJqOiHTGZZkcQX8Ie5qf5/xkhEKZMU/sYRPd00WLLmVe1jwHv0kFN4/jz1diXeNAKWZw8vmheJnwNPpU5AlfRxuX8UeeEdM8+PNPndmufz6fsprZJKLeOyvtJfL6fT/DosGnWxL/LkR1GxMuVZsa349zvtrBIzcQiKVjSnyQAfAXO2na347sgkUM/vYi2ZbCa4qyl11fwwJuKStbLJ5EdQQruxs6iYLVJU4F1aEF6XolqIitVc7u/tHRu/gnBDLXOo1zL4ItRfQXhmNmumbJrkMUx6UMSjrLyA/ZsT6+FSVRgEXr8fF6xs6ujX3eq2EFFdVK+SvCa9dk+va1yCLs8OknrO4FZB5PHTTfj39Un3nJjXfG/vGP4Lwn84oRrZqsNb+pXiUgv6XJkmubHsOIwgvGw96KjISl7p7QsXArarvtppubPN6oJNy62XpxiT85moo7E5JEseie44hzO/Ejm31XwkBeE194UnsA7DFmCr6VvxFfMogRrHPMnr02qZSZveNrYdt4vZgm3H17udtxKySeu9PTzawSp7rlMi2CTh8mpWZ6CMpqWoYHbN8NsoZmuMlfW8AiUVDguHrMnTItpft2a08eOmpViSzlQKNuUw06Q6dDUTVBYuu31Dz1etjUvdmBR1aG4UeoM97hOD8BSDZ1ZPd87i91MflQL1Qno2dZIJ1sjNkdcpShmEpxdasTun1qpczZNnau+OE1nCysNiIaNuS7fIQbilHk4lZFnkElYX+jiP/YwgPI/Dxm93gaM/12qcRVOXTX2tv2QmeckqfvDhAT3+WEya2cOeul6JrFgIR0G/XlaCHcxkurf31WFZ/Q5UjL29HM9X75TFwfSWZ8ImWdagUoSrFL7TDLVjmpOymNYs44f4mmnCgqpBX9s0SemY/chWNnnM4Kr0e1rBCW2Oj+SOb22z1OJdkthfkLLKYrnYNewaRVbCFqLsL/R3EK452loiWSUitY2XHI5K9cXf/e479u2fu4UnCO9bjwO1D3go8L6ULi+DUJ2gWZxkUh8f0oaJakrae5TyrLw3Mgdpfw5CHMAf/5l98933v9vbk3XVHo/a25NHeTSvihy3pLE+M17twwS7bG1leAdBeIcm0iKpWVHW8G3ZhCB8gfNpUCtf8JRmlpxzmlnX/GdJqVc6vbIl3gNRhAeiJ+YXLoWUiWlGM6lB+KU54WWZJrVSfPWMlzWnbURkZcojgcuMw3xS89lMKCWnXTePPdQgXFcaQlUcaBW7ySWfijX1bcu0qEFLQX0iCGlBpxUZ5osyF3YK6E48ui7Bceqrt77JaVXktcMFZaIu4mv0BR7wBX1/ZZTRlydvmt9mLKIiFsaHeR/m35znuYA3WLIkT2roOo/BxKA4pxLJxJuGp+tKF5MJLtIiryU9hMvd4RFXt+lUivqU/v7uWDtBJmqOD43uWda8FpvuIadGCZw4qLfc8J2CsJPkR+E5qfO6zt8/IANVp/7tbKeiMxfCKkdjjMoj493txleMWTzFDe2A74tt3tSFttiZf0dFmoqoZmUlcLrEpPDKIPwFw61wmqTi+m/oc/ihSeHYaPwKwvekjHg+3cKtO8n4TEBfZVVEsFHs7R37GXgkhw+zmcA+1Zpg6GcQ3u3bINAOIA84K3kNSqdrd0skcNVClza4uv9cHJYfMqUvwK38pqd44t+o7Q0qnrbIz4umVga7RRHxCV5ZFDP1dmSGqs9ppUelzZXWcjNNKqk/uHXfrAArulbgUEfIBDc0jJYUhOOm8P659AxjBxUv2TRKCymUdpLswB8ii8qjs7XUnzpoXj8vefzztJh9bO9puPgLDndYHrEknxagQjik7jjRmU1bY+nl9mHgNkvPYkcpwmzBq4TDl9kueQOcILRVsarISvcgT8QgvOrVVILwNLzpOTyF21p3xKlvHI7Up3DV4ZeFxOFv+nXbKZgIWV7E4sag2enN/uJmf4pDpzkscTO5M4OdHQwc8AhSgWfhSsDoBjh6wycF8ZpvfsFHXr87lULsXxEZWBW+QA2mteeOrfzoIbhX44pei079dSioGwzc9XSa03cB53A655SViHBJ41HUO0vsxhueTbkSuLg+w/dKK8t0Z5vphcRHDsLN7sDOmCyqekftRl+kZEr7EPalSlRVUbE0cHS5Ts/RJle2z5L4UN4d8yCwLOVBuEFTHsdcdR40mB+0ZMTJAherNeuV4XI3EbI+xcDnUU93HsHZRs002ntQJ3BoQXhn8PnD1sKyOR9QiirQDmByybrS8wy1PdpqYC3Fw1QOZh0R7xjzDz8NdTRQn62HE4RjpjCyBH/AGHrNZFJPL8MUbSfnD6TC/MMKSzjqAkML+FUws9Q82ldnEbYQFShvH6I+csi2dj/e2j7d+rWGbcfwOMMVZsoblnG5f6u/XyV5DCcdNm3SdNM94dRzUOnw5Iiz8nSrwl8yD9gsTioR1d4j60zU9yy6jOaiXQ1kU5ZFVYt4ra/CwFn1PXFYZlvTd8VhuT0de3G5mPFajEnQSeILxyxSiqhJeZ0sBIsrPq37hhNXIgjP2eeG+qb63X79xs8g/FBve6BZ3v/66z/99jsBFqCvv/3zq+5Xa8Fm+4vn1oEVXwEZ+PAdvApakx80CsIpiq9iNheHZCljaByvGpy5QduSxtDyHKPTWufYzuvOwH6urpKMHcyTWqD554ysqySfKUfUdesAQNshi1LBqyDcTYt8RrfZ5AnIgykP7AZ1lUwa+HJfgchr89T5hGYwfdFgQ1Mnz5JXdcJTXFZhzoPGP7YwLmhhVEdqfFW0LqIZCzRr3KiC0ON+OW7/DsKzrZGCVxVTW4usRUlL7puG53Xyo8Dz3ZtP2Au2e/gJ7ZFJth0V6brvmwE95mFf69ffDCzU+sRvO2ILtMkHod9xf4BWYbBxqs0G5yV6WekclNJ/0/fBHjbJp+tvDkS+Kw5LVgmMUgAD3Y+iKi6jRj0TORpcYjHlTVrfoqNSVu7ogxKZMdCCIQ63YQ2jRSsWac1ZDpYCUqsHOM+HFjY//eaQeFQeoS/d0SPTom4VSP13EN52zAq0bbBKFBWapS5YbxzWRvK8LkTE4qKGDV/NgjeVMlJmgud78BcMpUGHC4/29/bU664rntTy1QQOCOzNFntBH+JBUe3j3vHedJo2cn4zFqIEvYktdmme10WRqo1X3kedo53rsPSjU0qpIaQ3mob6WcXzpBa7eCCFyUNvAEdx229Zo72by31Db5FJ9Hk6+75Ik+jo6eYqNqEbvCy7OJ29PevnKX1Sep8xmeTxFP6fp0n2PmNRIePpTcNBIZss4xV8XtE8SeNK5OgcmFe/aM+sn6xZFtr2a7ikLa5S1LDEJ9Mk+mKVNVSKKuEpeGN6Rvm4ycorpr0AX9YkqeWwKz7iZd1U4ukyfzfsbqyBB1XKR6tohlVRq7OZMgJmpVbY4HOQUVGJmyMC+wttoEdiJvIaNhARNbU4lcgDXOyfJnmtNyp4yU0J/gIw981gRtsLzhJhO6zI/spgoT+lP6rhlXXXXk1jsWCTI/xEvUcR3Ic+ULEwPJ+Ju/1zr0sZCz/6rzsu0zpE+jhBeNdawNEJVxdGp48NDaIoRX6GgT1W5HE8rafb7x9EVV1kkzOtPlk2cn6R/gT1XjvxNm3r9vdq6DiWz+M4CH9eFgcfr6ZEtF/uwZzXQbiqI6z3uZiOsJuGjUafcfPu9NvuN7yJkwLnBazWm8bXVsxmE8nYrIBJBRtPyWfiBq3kC4ia6Yx6Sjm5N/hV4pRnsuS5Oqag+wxsZpUQ8RGqpoO7DSpxKgQCp5j+huFOQ5/CcdwnBuFj9flhrIO5Xh47lCC8P1PLbe8Ycqw5QUgG97oo2f6Z1jlRpo0y4qXFjDYrOLfcHNDkGQcj+7/2VRNLJ2EqHgy9OyvLBuG1OJlOya4J86L9EYRjNgGeHwXh45HP8tWrQHPRz9Pp91Kwpp5+wkBZKYskr29bur97FgjCG7amD+stmfJhvlyxmUrLfWhupZWA/RyieiiwQGQTEcdJPpMP+jsuec8yUYHZkmIU/2Fo31WbHMSNwWbeTUa6yzeNkHBB+dlKGzepkf2de41ia1rl/ylaNML7pl0jKnI0kZY0F3FdRUs7Ywc8qcnaJ0Xdt4K28ZfJoBXUFvm860KZNDplqqOqxUbIuiqOiF7ypHoF9LvmSasScAgXjk3mgSmiemG8fqWeeFkVOKnQBtLZo7pfQXiZlJFSxlovrlJxvtVQGLmjPqA1G2Kdq5/LJF93DX6NnON39JFizJokRn9BDfOnsxjadLCOozKvFljUDqJ5k++TBVT1/8DltlbzjhSENwbdJCwV1zr1nxU56T6ZyIrqaFj7yQoBBrmz9F5zVkkI1lFWiqRgGIYGjlIwTDKMww1CN7psziX6TtRSTdLH3Y8g3HRU+krIIl0IZSUtSqmuyGNe1uj3rzjN/QORzOb11eTNNsWfgDGM5UCcFM2AJvItnPF6dsSd+KyOy4IXwpUtBczttoOnEjN8Neim4HFcCSlv9PWaTnrQuQN65Duimk6tUCQKyNCPoBJTeKTaB1MJHs3Bn82qJhVSL2n0HmHBe8Oq7CxjEsI0GcUFXKS3dMAXWy+Vp/oD0FcggiLKyqFQsi/FIvGEko2EMGYe8f4yskIUj/qaDyr0Afa7QOMfdjLYBYrgMggi6536VaENo6hpwr8vj2RU5FO1BmhVhPSPTecTh3em31QlpslhqK0noKy35qU+ETywmjgXGadvAtfyhw5jmgjUb2Brbv8OwgdKzFJiehEZd0eUyD/lHMKw1xxNDWaxkPJsxvcFe1PSxyIfoekH/7HU35moX7Ws12Mxs9FCRM+HLQpErUTUVOBnxZPrGRDnVTRH9e2cXnNgTmf88IY+gQltH4BHQAHX0rWwJuADtJ4VaACNDMJrrpUBX6pIU3nbWbZAB0L7qTKZfzRwGDv20rs4roiXqKlwiNahY9P+gq6bt752ME60wSjTnW11mEryM3pREllZH11RcV9bLynua2cb3LznbFvvgRFcteUYbjHQcFCDlFqDtC3FMNRYLJJIrGwk2WXfOkaSp2oimfZaUDRZPqvANUhvQ03H6/0FEQ7v2EDreNQuK2J6lrBVBOFHpncPzS8QnuF4/Vp6p7hSgAZ+g2R83dvbTyCM4V5fAAJUjtgE/R4YsXeasVTOqzrZeZ+xVNTT7VOMoQetOMfIfscyISWfiYe2fRkDHlGrS2I40RQLUVVJLMi3n5X9pa818g67om2RviukNRtUA64QW+BV1wEdFum2HXuA7gbJpH6eVAWk7wneLtA/XaJSq68cpwbNBMwzA4Oh/HqobT/gQEiYxOk026ZOal7hEsvrpJBrTZ6gYhiVtYQ4lQzPJe81OURcKmOW3E/KS8bBHty8aZLvX3N2EdRTUGG7pDhvGgHpY7GYiuqJc1sZzpVjDzUI35GTan+9VW/AD5G82WU5Ra2taQsaavyTquAxxDF/+xMY0ZiQES9F/L+fKMRWHU5GPSIj8bn62Pgf/2nMtdejtsERIweGJMg5TMODfxicFilfcNbMIWgshXD01olp/gzCp9gBbaxqQcMgwkxgNGEtqgnP90EPrZLD97NJVRcH0VovujDJ651tSPTYNtzkpOgnsJ+Ae7uR4nG3vWY8m/Cesb6jXrFiUXS8+h2v7USv+vA1r9sSuAfh93WGGHq5toJpzUypILw3ujVT+Oby7VvH/KxgWl2AaXUF5/wCnfOuxxX4RgCRx0XVpXjBZt1PbKMYI4Zmj3XzyEMrVC1g3zm7n2QJ29/BXXorvtstuRgcDkdONUx4fTEr8vXkzZbvjLN93zDbwgoLl2JMBwWjuvbw9Z02IAR9lGaECBJ0igjZUeXwwWmX5NJtDJQ9ryxStajKAgI8NozIQdjM0JcXNbIusnNtWA9pDCpgIG2IEYO5GRkqc67JZJOtOaqN8mQ/8Hisu54141XfttVpqYoKyn8yw0wzk6CfBzgRi6p+d5rVrLmIjx+9ETyl9fx/8mSZy1YylzI3PVlJ0+MRmEetVASY1kUMnj/wfCU/QqQhrg8bfXUQN+cmFdf82iAYXT0Z2j3SVW8S83/852eWvqiilyR9km48v8sOwmc+9kCrLpHaMVrqO+PzgkVFBVvYPi2YF9EWmbMsK6viLyhsOg7BKbxTw4m33Bd5hBEDi60Yc4nvGXJlEe2LGgUFz0WtBW+iubB37kNdFZXdN+jcrNa7tAhUMWDtiMGG+AuG3/40LQ7eowhR0Idh4SvOqMgd0FHr6SUrdufzOP5BTAeiL/HvcYNBK/Lbt7bCogL8pilqgd93Cpk48oXax+AGwKj6mOb3ZvA4DJQFAsb6ONT/fzx03VHH8Netvt7FW1AeBN1W3ZSp0HeOKaWsyZM3jcte1k9rVAWyqDgY141+Wvb7GGPQpMok8iPpEvNmOs14fkbO+fbHL1lTwnnFjh45Nn4F4aMRW8TvilkS8fRzOGA9HFPcuuCIehW1tQuEWincphPX3lxHPfzsbXs57n4E4eEqrYsG4skZmuFXGrTZYGDYq2IfuJp1xwpzWrbJz4Zn+r294z4xCH9lqLwU5QvBRYmsp21gt5cZhM8MpqVtH8wTWcJcga+SjiFBeL2zXv9uh9RkCXZ08DJe7/n0uwyAR0YGgLLukVcALmYYyN+jhM93QeuevpcWs61yOux/VAn6LImv9zVrMM2UiYiEMlkpZ6V4g/YVjInqYBY65sxgKhM6KqPiDQw3bqIkn/m9naCQ3xqK8yYL+b0httKYQDm6b8nA3yqgHm1yJCjtoHtRdYGOIIQPa6MD6gAjCyhWOgTpTqeWg8mtPStoQhDeHz0K8LJMj1p/gSWk4iDIJXHDLyDeQCDt9UFmVA435HHsHj6ISbZ3NqkE34+Lg7x/nLGzvOCpzyOOyV3LJSNJku61O3OjHUXlyXgoU+FSwfy8v1jrn07g6/lQ5+aByeOOm8RlzVfYfDdcCaDTBF1LE1mDUWWnnehglnloh7E26G3uEorpdxDedaNdoQsJf+DxCE5gV51zUyryWQ3HREXXruZYlPWcUEusH3f9qA77B7yaQfJ+WcirbqKC+uNc8mb7UJKxCZ4dRM881ruuHe2YlanjtHzSxecDLknai9pvqWaaJUTayLo636VbmyepukpO6RRHlRsIWsZ2fEq5umZ0mEphPchxIabQ4nPGmQ307wvtQa0BnwErSjI745SbR7BY4z2dUclFDZoKVKpRIc+Z1mdWzm3vmUo+VV4wcjVv+GzRxLrmY6HN8ErP5QZeg14cUa3BJHpeuNqwa180D3sUg6HjjmYZP1R/Q05y/A6Y/E7ha4/qw/fIU+TLqErqMyqTnW3t7u7u3O4fYmjlhJNLxqsP9Cl3urOtI5i1peaK+bvdXzRVbwd0pLppXKcXixG1EdhuPnqraj2xJbRRx0MNwgvWKQFMg5ZB3TDAt91/6rG4x0XUUADb8QjXAeaA+2tvDK8EHy3mLm2Ycgjy0bs7+1R17KFC/imds1SEKRlwn/pOWjOBKEuxUH/hEi1FbaXudX/vmOl1M5ELNDMbYS09GmaL1wXoPe8xPGm9i/GX72LU5ZBP9g/cCyvz3kEEMWmPVPgfnd7V2gXLyquO4fe/djHeSyJBtEjfCaEfeMwHnBC2wKOho1VlZsPMyp0huRGcjs+H25jKVilyntaJkJ3dSdsSB2NQrIPVmdbNo5wHoEGf066FukiLA1FdMxJP8iabiEqnnrwPe9x+Km7TIoBgIhjBZf0OwvfrAr/HiyrQvxKUXsvTVKWKwixa8PQs/ZonszkEwV5UP2MMTsBOlcQkqfHc+XM5r34u5+lVdQikuAKI+qDTwIeKPk1yngbW2VAFvSiPqfErCJ+1+ALweDPhnCYtchDe7OJ6MFWBjEl0HA3CncET/vEAJwhvex1JEl4wHuQee/kpb/JojoOiSQshRe/iGeKKDRah5sOzztxieN9boCWMIEfOAV8E4UpeGj/KEvihjHyPIJys0hcFtuKCUxWHR2Qd6/V97BULwv9lpUvk+AEwHUO70nHXbTRw5P3f/kd4orjr7dwYiOEqC3keDk3lDvxZk2uli9XCla+FybnSnnRjJo/yes5gBbnZnX/d0K6KHzixXTT78FxyzozALUWOSSBwypRFU0XC7exSxplOntSHtotAyywS3A2pvorwDnz/v7QX2Qmmy2skhEEexmS4ObIqueeycbpWBv7D2j5yHxu/2nwhnXRNHhdLadJHaDyOqOh+CgO+aK/3cHsOCZyFF0wSJkM9Mo/EOiPHc3Q2sQzpZop9NN5ah/MoFVNPYzXsLONl52HLeNneMro3SJt14Qceagk8qHZBN92RDe9s2xKbH00q+O7iQ/tAmrN9PFfk4iAIn79NkyD81CduAmcOclssAM/JfF+IcsPLxdf60HdyrviBExC+XMyXpeQ7h2N3o9Hni9FMqMVVN0YJzGO80qf4jo4jetbFS2S8YiLFXJDXr5U75ummmReovygYBYTp1pIl9Xy9I3YRz8C407cQ4IIBz4UQRG4MpajAoz9vxs3DKmcR4AN76Au379PW1SF8lhYTnhoRz+Fy5+cmeUCvg2TPqQGpkuYRewGH9lodifEciPlUHW4DIFmsd3AgR7wCVQJSYJJMmidpiLTqXJ4ANlGU1600bwWnwXJUKTSkVxfapZJlLFRUzHGzs2gMWJ1bHrcorQM4pnUPGxdaXzta77GdHUIG8a+oDt7pncDhnK2P9mAJ3+hJqMDZ0hcx27Wz3btgYaBgM1MmCG96pCg0FywBt/rnfbol6uOmy8bMTYwj4zNBxgw+ke+i2+q6DwdVZDxNZvk6Tn0zLaUSEk5qp+KClKTzcQoJBYdtkzMAvVVIQTrCIxV0jokKOoa0RwvC+x65Pmkl3yusmjqxUv0ZhK41gf6/7FBxvrQYc0oJn4toH/Mz1MYVT+t+wkd3WC+P6jl0qcLnghFR1MfoHDQODMDrIkuivu3DyP767PmQ7cNPv+U1hLSWhOfWYgIQ6BI/+oEksX/3GkYe+w7F0bw9FJsAUHDYOIDAlLkj+JnHaNL6Ovw2lZYdhC/HWw+ZTnWIppqJqHI0E/rpeLTBcAZT9EKPfNUA4qD1XuQwoz3IcbjcbscP/Zmv4E0vIGIC03GC8AMUo87u6L/b86dNgKkPhKg+1LyoPoQAgla5N40jtsr/x54QHRgQEMATI0n5PpQhKGgxFYel0+kPJ+r0h68giCZl8iibFKnT5e9P1OW//On3f/zqS/bFP37+w08yxK9++AputyLN3+nygnrRGEeBq9Y/dYlLZu922ir0npGuUKtAGxFl3AkvHeiqP9DlXb0w06mUpT89Qlu/8KZWPTcbgEOmwLMeA5wakFGjgH9B/J5nDUdkC7USvX79cAm2E2Cp2pmGd1Ae1RyN6wj/G8H8D4bgv467H0H4xJLKeXr0o8DVDKyW1u8gfOGTRU03J3BGnh7wI8koFyUOwrOA4oSuVYRje5+xBmJIzrIutIRF5VUXjbOujuD0CVEn05Sp3PQUPFmnGSOk7MNdwK0/5JNksQUvAYLqANtQSgJPoc1QCjxxTQSss7GsPzQbDdlYP4+5z8Z66gD80HmTvXcQyVTkX7xFIHSH8KFWnVTgTH/tAmUOA5MhdJcDoLlpIfIVVb3reKU+GjeEWodqmG73zQ6TvK4K6NXpdDAoxlEs0NnJ8xgtpPLLlaHNfTes2Gxl4JVl8c7wvc6EEX2fAvMG9j+dKhcfQKs/3ZzyVHqYQBZPN+uqAebZ1sqL56RzHbIa6eMG0lrrV2ZpUexzCKiE1pRjjGvU5fYn4LyDaTePxOm6OqJV4x0AgCHgkMNrKtUxz5hjJrzccWaiVvdrQgtOxJovR2rr5Tsymdb3h7IhGeov8wK0NDcSHg7uBHjny7RCY6tOMF+3rMlK75R8IW44DMjuqpOMUivt+CQCf6Tp5vYIQdUqlzl0Dc9F6bFGF2UQ3rXj9+FV5k0JjgI4QKNFeMNM+gT/G9pO8bVtuRbpQueP+xlBeG/Qup234HfBsDUbzfJt0thrv1kb1PeyKmawRTOFANAnBuFDb3OFidEdCPTDtNGQ+8QgPF+liGIDwc04969AkDumOqp0CpzXT02f2zIr+pqVG1G/2WYvIKHSJW8R+YpFRsiKPvUl+9ZD/XiAuuWl+vrd9faw6+1h19vDjreHbaLat5wvpuVun5wdKvJFe4OFD+eyFeqpfBpj+YZ/mCfTJWGAP4gUfQk8DcI7I3LfIRDHaVIhAVmc/pqLw1/QX6A5nWuLWcUiFbX47V/hQYHVQIVgl6t0gxzJD1lcZF3ME63gfR/KmHQQfrOqt4b2lSHg+cdB+E8ndGuYAdPqRT+wQv8aSDM5dklB+MgiAf6sKwa0ILyuaEks8jqpj8w4z3V0obSOAjo8wgrzyPWtSE8KfAtF4vpgkvhKjw4270sOnDyYgO60KbL6vGgTNBhKZ7g2fwYheYHoODtrwdDJ0t8LiKToKHWbhAHVk/EFUkKyf5InkC6a7XQgHUw5f9/LJpBbdDqbSEwykqczzg4QoAYcT+B1iqtu8uGh4OYweoDcemn4q1r4AFnHSeEwKDQTGJ1TKgboLYXNDB0awZzYD/hm9MVe2pAxKiUKsGJU7roq4qRdNypRZ7sXA1oXjJdggEsymHoGWntdoNP6ybgrDN26+ogGcaFbL0LbiSXLrHNwgb4FL3GSgKXt3ogrTJMfdG6vdmd0SUH4azcFDEG+8jbB7HiUj8GI40nYLfXxMskuG/pmm6qtACIly5JDoWb9vRUwAB9Y4a8lz1v/T0dq/X/SckysOUGwGCQrdCSuJoNnDhtOeC6v2rxozpMcGt710ZVLU72Bm7ZIaySn+76k8+DEVMD3jqUFFE1UU70otB4+Ybw2ixKED3syKqUVciMgaAL+D0KrNNkhIqJ43IU9n+ROVvjk7rty2x6hXZ+dgeKK0dSui5RUJhBkaHk7CcuO4SDceOHAkkyzjxEeyjMOCw3ZdKDiY98wua2KnPKJSK0rkIPVsiBilhp8N9csQTNNx7S1mPVCKgjrBr/T7REB+IjWLD44PFJ+JCqdOqlC7I1fQfh6md92nP/AYk+aqfQFKxvR0BSSjQHnhYTF7KqPx+PYza7M5ADawu2RQG1YbpfHP0PsB8Q/b41KGpUzuso7q7Zo/xxoYfp2HQS/B54WyveJDeHAJOK7Y77gnO0vrLegQdTxIAU93e7x7EH0+PjcYHKxJJZt8Umbv3/DpWJsvUq0vdNPe22LNVLW64btdabQU1rylZN32ha0uWx6kbX/2Ao+7/6+O+BPNoL774yKoCYzJIEx/vcHuDhxEbSFp+nQRVS5G57qCpTdyzS27aumCxuOx5JqBW46dNvXLQ5r9UHYQEfHfSK8ANjgjBy+Lr5+wwi7b7VWYl1GVrR9YBKvuoH2mB4gjm4OgI5R/czbFpdC5iFFA/D3y6oD5iU+2mX29gCmdsFTkdc3fGwOtn+ol+rE9YOBOI/vKGolEHQ5Zz2ow0tO3H8pxH7o0vDIWFZJBsHtnTeiq5hjSyZTSyjwCnUSbYLuFUeQUt0v6IwDSCDEJ7WmKIDdyVoszzZJAsyksKj3UT6pBHCSzdDgDjaslKqJGJU6MFjvUZtXgFuqk2ugaUHY1RNQ6EKgBTnSNicIP205tH60SP9Ouz43CK+iPwLNcoipAd8/K3WCgw7HaHa2bULiEqY725dbgoajOSrFmkukwi2nKX+iEoKAz1VKwjQFMzNqz40UiMgDvtZfDsigVXCQF4TrJg8MCtBpli3eOGjEWy8p8mNh1PcgijxIIGwESzBd7EJOYDkBLfSeruQABzg8Z5Z1kvGWBE/gQlthRGNuG5khWzHB/OQUt46YBhTZQVdDs7tUgCqXPIn9RDPuWxzWF+xUkULWV9o0EbNeXUedTLuQFzt3JD64on9HvKbqIaDp3VDUrISvQIfGqCpN615m3JR2QIiB5E2QR05wjD7WFKUPa0D2sAZuDyASKLOyfOjh6184+en12xEz4B0GTRFfin0FKjOkchPxXu2EGFKe09RGcoLOSAW0ezNKEiWzHFaZFxafImei+lCr4u2Y4bUgNJ2qWZ2if0SFi747PUhi8e60EoflWVX5WNnV3pticPUVkaGdnGzoTK1sFFNjbPlsn7SUWwYDFQkNXIU2EXnVZatml4lO8G2lStO/hETQ+1q8T3mN9nuNsGKI3zDCfHTgVqs43omLZpKKfngZ0V8DdGo/eKeCEIDqf3IEiVOxiFIEaW+PbOURA6fbRU3Ybkta3/V4xjvcJ3zvVz0isH5pNAhPyBIlD0DQ0rK4pnpWP1werAT4LkvFJhgQO4abC0Zt+XEPl9osmORnBaEujtwvkTFWPqODzPC1a89VN2whdaBFaKCutKiC48BgZZfUScGT2OzMza86Mliiz1ufkxS1XbQb7/mGReoCr/bFURDa8riIPnJI7Q/Hlb5tsB7a33X7INwQCyWF+oyOY6fZeQz/BeFNn0T70v5DcfWRqq4SqG7qjRCbVStGiCnBdf0TMx7VdwZfnY5x02489VLrghLAdWieXdXQQ7Qy8/pBZGgZBbiWOx45WA2KqQbcu2JKtHN7y6VmiF/qlvNqGUH4KycHsCsC7uK32MwObUbjRBu/9OdoHsV0JOUQKwjfxaTE8xhZJpvpNDnE1WzNl17IjgjKxTa2mTsOqDWw+9/x17aKRRvMcNMvEaVYxfeazTWCIAysa3bEJQebIJoTAPJJoynIFyskQlIzpYRtr5Q52UiI4sOKVcXB+DW2D/iibULYNZ+O4tscj3CD0CiqwcC3EpXZYpe9VEZCCD/t+BrRBePVVfAvzzxl5DDKQ6WBChWHrdf4s2CKVMkaPJd/PlFI3aQCHa0eDqv7l5N1q2U84Yl/OHHs33B84vcnDPWjyGrZTNwwqh/+qg7zwtPlZ6vGEFIgmdN64A6XRiC+7R2u2KH3Dv8LdvnTeKA/+8xhPV1ajxKhoT/e2m7S1w8GAhbtCKylUohgfWNACsHLXizJxe6RHkEDHZrW5Fj5owvjfIxxnHDV04wtMsQBuMTYYkoZKW2W3wcMjtuUsvIhhOg1eVRP4+n2acjzxkPPB4zVqaq2cxr/hr8+MKIhoeh7M6mnO1SYcIqRSelp+FHmUXlkhEpWBwjMS71t9Oh1dVTF8JdR63xfHDEK6DnbEf9SJPkrO5iyF0ep0XVAIU8i4vAJVNGLZf3LFRtPIZXWCNR8mwjO9mJ3oRFCmVRoO+sshhDmghrhORhDQSYUeH63//Xr7//0GzyMf4PK2+eA3/s9xXR9x8sbEPb5L3Ba/X3+h2aSJXUt4n8tqv0vi1wg80tMwqOqmX+Yw3z7roibVKy5TLrKBQRBwz9V9UKD8iXFX5zvKISSdhEIn5OqoVptGKRv8mnx20qI78BrWg3jA8x5/BYQ73DXbyH+jVraVxQHHDiP+H8oE5pK/dKKKgXIQpKbao0Pp/35HS+/K2JhJL9BkLPzM/ysha4Hrep7FUb2paBkHwT0GuQF4XfduBG/7qu8ro4edytDB18Vsy7ml2rgtpAw7dK75fRmjcJDDcI7XQCyeuV7e8fGryD8bOzZ7e0dj7GD8PRBJLF4kfzFQSRz+vOe2kDbjIg0ZbneVR9TCNEdWwbu2IH0/7MtYeFK93dofGQ5BEIuRJXyskRYA1xy7Qu/v6BK05dVHKK6P8qKHQAWhKSVbBxYsBUZ6MJC1vN30Yr0u+gf6Xpd2CKDYBpG3pAfTEMJDI7B6GBoDMu6MAA9hrpQIp4ueppLvwtLadzpupCUzGlVliA/lRMF/+uuDelFCobRmaHd1YevmfHS0TTca2aUG2AE/2Mw/5GsRba3x77n9fyLP/+RFBG8Ivq5DHWiu+4QDHv/c0EYdhvC5ZUn++BV54Bz0w5szqdmLP7QslYJqlnsrGoDKQuMtVcdTVnoNx+sHuJr/tjKeDjgJUCZZLwEFdhJU7jtJkforlTuRS954mOng/s+xTYzF6vXr3U8UxcZBbk8I5g1ZuRRfVQCaCZ50qzCSBCAz5rSHVBPBAI+3OpJZq3Yp5v4dPEfw+T1OHxodgWIFCrJ3Ons+6EbGcJVt07RPdT9IHxiXndaVAyq3zApZsbFWSze4OnCwuBBuzsguOUFBPHNnO90C4W9niKK2knrrgXeW7hCC/Tpvv5ykVR1w9PNGqJMN/VZgT7agjwVvaXu2GIH4WgnyXgnie7k9GKa0LHjHz3YsFSyWKUgtSsQJJk0JWnWELvhrELPvCCzsq5Y1SZHkpaFy44JPdDOnyZT4HLEo+SaiqfOO9qxGpvWBgRKYhM4IDhtLms0JQQlqgtEWroAoIGAVVIVBSFMXzJqlEIJKKA9MmgUrqgqf8MTgnRg8G5WXF435NxqQmtG3o4UWLAJ5C4Y5ErETSQuWpQI9JqHBgnSeKB8UwGpcgm6dOMGU0fkbUOs81NhEg1LBZ/e8aYOAZJBMSVj1KYpISB4ViNMMXGYyFqaw6W4nvO1JC8x1KEBt90FTYBIEaS0IuhOPKrFGU3A0q/wZFKezxpYA3LSMuFgqWNhKXz0hgNtZf286yltr1w32oy7YVV5s4q8rVms1kZxv+azmYithHEyV4FHC994VJ/GfH6YpzpLSp1RqTKcypIq8dPbVTBaANM7S6ZYuvAKkTIe74ASg5Wfkvg2Uf9SopOXZi0eRCHThsXR1QH+hkrKSqat2yctKkyPUyyl/ZYA8YiLbsUPLikoLxPM+bqiQXQT1rmTLVCr/JUsyornhTfN5niYGYSPx1qqG0EyJFAtFb3qZIZpDLWbDh3jXqhibVnIGy7X/HnFYZLKtuZQlZX8C6M8sZ58EQAqqGDmLptpSCIIP1nah4rap0MOTzCetg7CXw81xFSukUsTAlv4ckn7gcveNZrhe1AFqCEqFFeaIHxkZM/hxyjbj1E5T1MOcpetLDt1PZuooJ5uuBl2xKVZcNPOtDs2fwbhRZsLQSc688+GiZMIJb1m8do3fUuRSYNTTpmsiHmaAOqfTlF0yzq2S4nGketn55Gj/743Yw4z88ALQa/HD0aHzmcc1z0v3656dMsrozJ5gvCOW0rOKS7UYew5Be0UDslFtbOpAHCIAYF69Lk4LC9i5hQaOKsj9gaWR1WqPl1XUUmYVlcXqiI0m3QMyKwDBvz/icn4WDE+dhm7irHbZ7xQDOcaW4qxpRgX1B4PeZkLBHQ9V/Ec8rI0OMmno/mGxyPcIOwlFPqS+ba9yXwqJfGqr7hTJabXHPoOOzyU+KLPtRyQPpRnjd87zPq5zeQlK6sPNCme/rZLXyLNfW/PSGii5LoKDG9jprWH4QfK4A0BtUYS4PeG/r+3d2z+DMIHI0mAXwEkQAIxBu+XRQlYr5+NIyoej7EhMXG0dQn2eIiffqfM6x9VniEoxGfpTzCOQ3zElS4DUZmj4c+rZm1jpUCDfvq7vzIZ/bjjBeHvf5LMdgJ0XIj0aLROuO5vKKlRpy++WrkTXTPB+PH67QsgmL9WSp30p3xSaU9VZToIf3Pynqj6dxCu9KqHB0Oqr16Fvv9pXHLPXmPIUS7r0XJpyzvE8JpPV+kjrguud6kqkQ0UrGzy+L+dDK/zhBmtPjDT0dGP9AVfzY8namtH8riPdDCS52TD9n3so9/HaI08TO+SrMkRf07Eqz4Au6ef5AE8sRKKKf7TTTMmahDe8pUxyVtQiTturvGxTQjCP7lJw/YtdNwlt2AL3jHsYSrqRwVmFZDfWhbyVytUIsS4ESw/mCi7LdgHbpkJzd3GJmsBASj57QG2zija8PPB5nvfz2qRJNkk4Sp5WQlZec623GU7pZpgTy45RLjqdTO3WoenqUPqZUMvgQAzfEMvxwrSDPP6mdlGM4SVCMLP+tnbiwJMBCmrvNVuDHYQ7gzmfnsaIkcnmvvawLNZ63OBvNEnY/hdcSC/Hqq5A1XviirrPoIxmSC8aOWeY/z3L7IJ+ggPIvleNqnSgURzgFP0MsAOfBXKxpQQpWFDIj8AepJDVl0lWnxCMhSLVurhaP46gyzFuIyyu0vEkoxf7kTguEdwVEY7FZSOzwTqcUE4HljzNwwRqkNU4VGCV7WTJQ+PwUqfN+CeIVm2ScWDLkkevh34H6Os9XDp2g7yM/ZLEf6XrYpJSRZH9c7LG0aafC/1/mNfDn13PTZJ8Fyngiz0Y1/3tQIL4N1eYVOwxqKaFStMx6t9kSSvt15u9uiymdAoCASg3zdHQH6Bdo4kbngamCLK7kRzh1L7W8ELbbGbSrm3NgyKM0cuGyxMb6GlUSEJjFWYutMXwlJAgpeQrVk02E2/VpUj9JFXiKpWEogWEimSntfzEOR5ioHcwpw5CkdBfVBJPgNQ70nypsF0MMR8wRhWRAXOwbwI0M7yLBLBPM3hr2f+wgrHPjImr7tI38cuKQhv9KXaDUruePEOoKK6HwkBsmvvDrZp+70+WOsLk+nH6oBJnQJ+d1AOAA1L8Gg9HBWBwwym7A5XHgOxTAA2wp1RGUg0GR9PBBh9w7c25zHPwL2Fj2d43PgooT+Vxpyx6RQy4CgsSy2AcIq5aqHIdVrYhkW30NxtpHcDzT35UfhA6Vqkd1h+Ao8AhGFgXpuJVHDTIxgLfR0FCdDkVPOaVQf7Cxq7Qs1WKUiwvWIa2872Ew/EvANvoajtxfWmhvtepyZd83CTHIJubns4OnMAbsHHR0gMdeWLfb5T7JnAMIBxzw+BAcmcLVjuHb9MWRUTQRPnul9iImZJ/tTL07PBIa+3mBuQYAgeZpW8fpYYuUpib/H2W0AN1YEFBwG7My7ny9Ax4MX/0hRS2fE9oZceIQu1o6sRoELFjwkHI3wxKuWCsDBeB26DnQHojB7Ox85bCw3dyc7Hnq6WSKm+rGKC5TzZ6YODWEUTMOlRQbQ/sBlp2mSecVilFbAe+hL8kYwn+dB0sCBNdIjHcZ8YhFc7XFUztiK0nvD8iC18r+GJJdTkRw0HUJpcerBTtk1ZjbJCpROceaNSMab5Jys3AWSlvCYUlSNRPbcaiizjHSiP5zbOm+LgnbCesw792PVMi+cDgtbgOnD+rVXEoWJBez9frtJCBRsaOB4A/KA1+lp8tEonc8jIz/fnRZU/XlG+rMSTFUVBEXjkld3qrS3ey28b+SvdpLK6JAAofEc5r8GHr8qRispE3HFS3a720HBoVuh35UDWuMU/WoZR3sRtQcXY06L+wpLwFSHxVhYxJLqVd7gPA3QFsuc/W9qAyqRAAIWUiCmIoRZAFPGrt2w94/W8bfz6ZI3BYzHjtdha2hwxTowudgZaDD/Y4cowenB0mGG6cNcnPnE/zpHNfOLh+cfVqpOOLKjvVbY5yAOvd5bkSwT4YeAV8BDvewUV/jGThJt5yxIicADD2dxDZlKITc/HkJm6KQzS8Om41Wx6QE45HVuGS+a2T87Ds5/ckED75BwBD/ELH0CTVaPneIlEW4NoABbK275t+8QL9+Rr069RTLLGY3HRpgbH0F7+js3oSfYuamIiwT40LfoCBpZUEh9KL5hUWQl15pE3fXxAgiwqPhO9EbbHTQ1WpSSSQtnQoGTM3h79TVvExrDERw6LAvvb5+DQ8RjtlTcvt97K4FOiq8Gfz2xG/1oGuUUqNBCzZm/4ukPcXyisSY2vpc7Yba+a0KIB6iOdE5uWoumtFlq/dcXQN4BplizRx52+DCa2iU7wqi3YljZ2LhKLGuI/UjGtaT8GW5NaIWZt8Wc6+5lVpPo8skMl9fxGX6KrTnV3kNm23+yLdFjyIHCrEzAqXMFhKQf2xjBboacRThlNnIofnDepoEldNwmkCrXC9/s8MlkqsBT0W1zrCxEyUqnG3i3jbeGmCkAbbjhstWXQzuoyW6satPUzVXjzA5fpfTdXHCmquOJSrVLiNiiairOiHi+7EvAErrhEBMVbd6m6yMw9h9F+Wy2lXSYNGVzcdC3VFrfQEJCtTx4ustHjt/i9ocmiiDpnCDrMzhqFrn2myvbBBBDrPgF0ivkY+Fha3E0NQxAtRNRDkANiEH7cIcT1YmeGWEF4aX+B8Cx7e8fqryD8cD/JEra/w2RSN+9jLPnr0EaP06GG2mtw0cWc47Vd7z0vcqQ3efKmwUDjeRB6QeNAEhw0XsA5LhmG5QfOeDo2IbGcU9y6IJy7yw4CnayrILzuELEOPW5uXfeal0jWABJFhFmMHi5GsRP3vgt3l1JsqIRC4foCGxpPDoD4ERRa/dUvd0+7BFVvN7HrHnrlaoDcN8VeeMUwwaZMk4h86sv7zTD51BDc9orBpw95LbmYUZq12fOz0SYUo70MwA98jIbYY68Qj+Okd18XHNFE6hkwbfII68UrBQjQD/VTmCIgVYvWd2z9DsKrLbCfWRGSp3r6R7yUCi6vElwWOeYeTKdFVd/vi2iov072iiGEIbcIuaOREanVNAEN4hQWLUt+FKdAuU7LOT8P83KabbdAfe8DQfDyVKK+o3PJmx0INURLNMQb0u9WVzgPcYUqYITBU7qgv/i2/N1HbTJInMioEjWVUa2SSQPvH2symvb5pyvUeMSEH0wDGhfG3CZssapw2/PDcWFKOXr4+hKIGZhGcDfPklznrrBvalH9vnTTzwgeEoQfv/4QH25GSuN5/NHFp8qrHeAjODAB7TFJ0+LgHJgTEXSPUkx6oI9vPKCPb4Jwq6W1mo8rbDFMmEh4z6lQ6fRWE5sThLeQAxgPbsIkwTi1bMqh0l3xqgrCCz3khrNIQT8MmIIdvEhEVDQIsMB/2FL2xRHJoy+WOrhrYjVqXL+EqrgoM+CWKSIxq5+lycTBgOwYQbg7AA6pt1I0yLi+qHCgEcGFLkS0q7XGSy2MpMKJnO5sX2tpWMqgmEJSxyFw1lvOTIDwmyaJ9jsGRQHrur74IV92GR0KpLogJFSWIj5n0gqeEOJge3FQjMAEKlUNVUHrxllCpgRnGrhMPlRAlXLOS0E9wkBoL77S/lYaaiPrIiMcTSinJiqoUxJf1wRYWZNZUzSyBSu6YCNa5uLgmgFoqTP66G4v9jnrbSFXuCUNOpj8KK4qRrZQEe2Hn+AfJh0D2on+bZ++26N/PCD/8YD87oD87qD8Cy3/wqX7xk/03R59e+C6WwP9bBn9rHd0qjqjB9Rj6BbXOoaKg9dN+hzd5oLidPCnl1UFXqjU1xY5uGmV5cWtDDO/GFbGWre4RllcXbGXQE0BxvSmBxC1+wCojG58lLPaqABcX++RcQTgwpY2mmpUZJM+mipc2KR2aKofavTUmm3FW97cVWesAMUKejeeSO/6kVQTPqNUsenO9lklkm4nGduK6RMHK+a0kU4FXrWmYRE/TNS552PrClDqpPLUJ6OKnIsFwiJ31Y9t7FT9kYqabcde1gxZ3prJXaePBtBfIU0JMVFVANv9ATnErtBCGx4hpb+teVi5OPDBxtprFoLX3h1An9V1jQEMziNCWCQY3zECT5sohNqoqWyMWUAHg5SI9R5VlXr+Yx+ylk62Hfyuqluzt7eyaNiXVLeJqpISh7ablqTnsdkvjWJwqfaHUWjavme4M6D261cbkMH9JhhO5auGvbVCqWsbmTfstUCzFS5YRiMJSpcX4xe2MJgRD/pss0I22P5Tnm8uk1p3BZDKM+EtsT3TJbZvDHFh8Nc86MMk8h5VJ7xMaAFyXyIuAG3cHxA4OSROy+uIiF3h6uvKXZlN8qmqX4iolDjci0hVUT6pyGf1PCBTxmZgwwZpn373x7tTCE17d5oVcbqFuhpp+wCYyFP7eKM13E7pu00teujBJBmEF8iqxigUGpbmSyID8B+CkVKQPxdtKGXYRFx05Rz/d6ixwP8v21TKIXaJUIEvXRMZFTYH9RgwoWOMFrxugC/Hstahx5iZQ7jKCo1ZyWBMgbzSVlig4oJi0syCEGqn1z9KUZ+LKagCVOqKV0e/iEkNk0cZ/IlZQ0G4FifTKQFtw6FS//2BivqEsNNLXUUY6A5D69+NjqJUnIZVEF5v+r1xaiFbH1knT4zppMGIMe4VFVPsX2iEUY1WEGHpDBT7VrHqKhEwf2LhBYCtq2RJysHrDkqV+kI0Y8ZrDflqIiTjKkNnoRsettToyhsepop0XfewIHf/8FsPo1+S/ngFqSD8bkiK6iPTjjHSmSkWhF+Pi2V1ubSrrC6XjkrwWSp2lnZFYkt7i1FlW9obiQWhDzKbPiAINUc3jO+Nt3ja9weY2m4PqLnSV7Ead1OjanULzC1qCyNXgyL76wBrpOjOEjZpYgRSxajkmMAhWpTsPm738RDLQLsehASvZ7WGESYIdOXs0qmLx8PMIHxkMF1s7yavkxQ17WIajsiVR/VcW5KCMFguSejjKwiS1fv2mKAQ+2MdURlLlaC4VJBO8uNPhVLJAHslCB/35Lod00mqvWOIcjg7VCLnKj5ocsRqPnvh4JqrmNUW29mhG7djYZx7iNdscPMibvEMrvQ5PI41lZx0uiLRmocK5bf8uOOfXffhiONwustioQFSP5IfRQuqrSH+MHJD90OFzsDGg7cG+d3yaY8Hhxgb+7slb1hUZTomo/AvTZYC/NZujuNBHoAt93mUM6KBwm97JWCMxL/h4eM5Bh/nOF75GPc8KYUtala7xJrS8HkJKC5GY/HCnqepAXtuIa0jjnmXYaWxzheJTOr7riBGYu7aQg9dofYbw3cGpZqgFOgKYqAZfzQophaTomKdxdt5M7Z8JabJ4Z1BflHWaHOxZo1aJtoamYM8552r75RC6qfZ9qbJzOTMfs2ZnFmPFo/FtgiS9OARgLaDyze2Pr0bekDi22/4mgfk/YBXWVPe9nCo+ASF1d718HGvbbLsiFzC9z0iPZIxAg2MrpaQqwYwfPMJub7R1ukCxuPX1HzyB5vu6Kstb1nBAkuQQPDbwBmVzKiyt0oN0xOEz1fAlI+LdmUfFyc8+WP9J3yUy8VlzvfF3tvh1AOmWC7LQoqt+L4Xkx7bIgwOOl9ModY130Z2TzO04N0whGhRaIPds+yaCU1v4ODrOgBYJBlfIVq4MdbrEnJEHhn5uGdRTQLv+dbuzoutU/rnP54IE74WVeYgzP3pRB2pKsAIvsGjWlROr/98ol7zIkfE+h++crr73Ym6ExGEzPfg9P94os4g7ZYDBpEfT/+EnerHp+wgTqf//ld2SgAp3q5P9oL4wb6/u5NNR14X7nT87mQdSYAU69di+P3JelPhfsnCvc3vT9Tf73/4kn3xj5//ACUZnA43EcPewnZWqPZNLl0BwmV2erjlCLi4usReBTqkLnDqSF1F6xdkfgKT01ov9xDPvldcMlKv9aixJEySW10FBYwgcsa627EhF129UkR/ZZAlRMYcx5M/W7mEAh2g+CEDvKkBXB0DHMfhlSe7Tptw/9YX/Gi0fMO+OFJJDeAs7mozjMlDaMR4rxAR0GkTy3tVRSReX0EpIwoDfi+vFoEv8NRkSsiVZ0DFe7mr0r23zDoRqB4nkHTO0yOZSIwctElB+KteC2Vydls55CD8da8lr2YNlgXtNe5zgnD7bWtaBOGO2QR7+lHgAU9qyyFoXRhKRaiXkDw80oa+OcVUZnSdCedr1hp4Wil4WeGTUVn7MqFPFk+8eW1LXmAiIxx9kbOjRKTxbZOiMNEynqST4hDVnjVcFEpmhAaAc+QDrPFBRTW6Kh5yur3OGNaoZROu/RdynmRB+AGi5UbzKm+gBkgFtvSk2J6e7/5mKaAg/6Ij9Mt4UK2O8x2dzPVXEJF2gQnGHaDbOQY+ccjcgkMIS08xljV5xsszKpAJ9czqfQZK4HT7Q6awWiDE6QJjU8CUKNRNihgoiNlsUcqmPohaygeMAlYADhC4M+FwYwwEEVV1irE4WdTTnU/t8h5oh0M1X4cRGlU+wMEcYexrLmQt4l+vUhoEkGqbnGcTFRlSNpM0iVA4Yv/+7/+8/SsvSEBekO35V22ixFfff7ul0i4pz/irf/525z57EaXf1F999SX//o/sMNnaJQl5wHEQk5/sGtPhazyA0iVQdQP2SSpecuySgvBaX+pfeVJ/nh/d//rrP/32O1Hzvb2vv/2zsyzq0tffY0UwOBHMHNdEx1hy1LMFnx8U1T5HVDC4BhkzcGWvmyrvbAOqCldoiiujDFmiKiHLIpdCqtPPhwfq+QD+yhn1gwqTfqB+icOkDg6caHW1HLcOI93vbVcQAfQ09ksQPnT5Coe3C59DR+SQmOl84nL/kSvGI+JBeIT2jBal3ByQoyIDWJNmQACwQgeZANt4c4BZiYrn+9cHuLAf3xngddWjh4bdPqyhLrrHNDQ8UhMfDHDtQtR6WhBzDbAnWQuR+eYThmHfRFb1hSnkDsh3LbJsJjOo0AI/KijYhQHjlghERtRQ4xp51yxeEncdX3E5SL2EVF0KlWi3kWYE8WAflEIA/A2X313+istC6gWkwme1rS572aTEB0S8Ct5ns7aRqkG05tKpWtElIOslRy02GybtBwr7UZ7sSzarq4lExZMU5WJH+VrUWJHpYucU183PdZWT/gR73sXu93e8/ByiQG52pK9F/QWoH99hZOMP4Lpetyot4TDponZghIpNoqAkgHSv4n8bqQ+klky38tHgkmkL/tHp+Sfp9IannFBrr9xQ99eliLSsdc0CTVI7X0BneqzjtRJZKuxYGe3t+aj/5iM6d9VKLLsrS7BfLKet+dRW/XGK5dgC/TI1/efbK1Nji/y210W7tUbeLgClIBfqOIrlHPy30uF6D9yKLdDvgP7mVTJUOMgWGCqhNOeVGHwWtsjAGKwSSr4xtAIDHUCqQT3WQSvg6aAF7h7qwBLod9CHsnQ6sAX6jxHUQjLjDT5GW6Q/hi7zdmAMtoBvDG4HnjGYIv0xtDlU+wNjsAX6Y+jyoAbHYIv0uxgsEOTMh7ZIkLecllIxBm7DFvDWwnI68NXCMkQGS3oZVcH8Jb2W1gQbuwlDwCmlNV7WSj25612bSgDmXFeP7fV6n4eWP7d+Vme09F1I1c9y2lASD8zFkTYfdW26bJ0E8hYBLpWnanXFs+9byOdNVjZvJ8/eZjxYHect5EHDP3obeSElfwv5CE6WbyMPbom3kler6mcDNdKUHaWjKeOkPXG+fMsKa5SG73Ty6mSd0Lz+vGuMEZVOpVqjWfcJ2iKGmrCkwptaTeoqWSQ9q7UxkC5CUZV5ssMT26IQr7p7fm6Xq/P3wJiUhcaG1kXjulf9fKjinFtmsGtsc4y3ydRxhqJ/e/qGqQM6zCWdGGtOvxPfvHI6cbGfrU5s5qvBTtBKhtuVpxMK/71nNobKB4AF6hZeM2UwyQnLAb3qKHrmqKJNeZxCfm4FHn7Mr3ZErYv2i6nhh7BufdsJis4g8smqxNZNIGuutXPwdyi8ihuD7PtJPc8EVBX2Ybp/vnJnbSCQ08Mu9qDsRg5SIbq70BP+qhOARlSmTikmqphqd4EuiOt5W0cQu1KZCpiJgGd6dF3cHRNQEeEg0gYlKwswRmSrPa0p027NbgXTZBKVJdMHN0gvcVagfx6rOker0XjpOVMmCL9fVglvaX+F3eFvRzucLulrutK4klXGlbzFuJIlNf+SYrVultxeMu3K/nFJScnnFboCpIOgw/OULp91iSJQ2whYCMt63Rl850X1NvX6cM34zlsacHQ1EhWX9grULrxf/JW9wad0ydPHxKadsAgEXUpntjse8JNdwj1DOpewn85v7EuYj2iF14Xf+vpQF09tBrnjlPKd5E4vz50ijU4Gi32MCMJPTfGHfpNI4tkK4W3e09kSVLiS9lFKBYalDKxS8u6IDGGUbD8YE9lm86JKfizyzREpPs0KsTEscMnDOq9pqaB0kA0j90N2gbosn8bDrOiakzGCZS4J38HDuWLSKFjPuTLYf2dSpfeV9bU+S4Vjb/gKa9KV32nyuDira2xSfajLtVR7gfJzYV41VjkjZ6jKszvflZqkhheNSpNZImWSzy5ZhSQJ8wnqT2ZWT9AQq+kw9E9DbsCZjpQWs/vdr0QyjOaM0gag6DFDibxElzshglJPmyw3+hHF9EL3i8cLNDETBUIaDooqlu/Rb3gkRgLSRevn1kuWih5pgqVCI+vGrtZVkgEGB3qQDuZJLTBE9ik+bpkgqiSu8Oi+hkRAgtsA52sOyZ8Rz9faOptmRRS53pLpLclmQskQaw6DqPKaU59TIbFWiXiPkK/ve2p09kh/7lFsS2/fqjlg6bUFL9vFPwlv6JZFVJ69Dl/72Uj5z6ODuaisOqDrsgEANHgsWEm1Dab5UM11/NTPqB+VKAWvP2h/8Vj//ZciyfXfkK12StbVFAwR8Aeebu+o5DlJSDQs4zmfweAkwN7OIihiiLU7dTkMrAO3YNtXPeT5gm37xOdse8NDxuCQJFLlSQmcF5C8C1nrogafTJL6PPGTOKqpso66xL99MfkC/fY/fP0b6O8DiIxmlaibXDwaq/eJJWpoot1fKjcT+YNRIe1JvzcmpVI1Ho7JdMlOo12pEjGnZE5Ygx/IOd/++CUl/krEUFQFIaqkqJL66JQU9R/nlTjY6NcqpQzDILzmsDpUerdRLNpmP1gs+9PqWEs+LVtwy1NHtIaSn74CpHUtuyKTvRbquxyRSGDrKMog/NgjgU5r/3WRFYT3hlu11/56QIZJvhA45Ycv0ckE4dMV+mkvGgwJo6ZnCHrvAPzxhsx9jwy55Q2hT71CFO/jv0HNhW9irG17iW88YmqLqcQskZhN47uSIxSEz1bpqb3urzzSmairJBqYHYoZhA/GWo4+OoqKqrj/Aprrf3Rd2/YS01Ex+Nf/7fJlDlBb0PcV6fTFIZZ/DqpW7Q3s+GSyycC3Axy7tLDdZvTRdzWIfV1rrv/Rd21HZw/EiVTFUFFlYvpnT9uy7f+RT8oq+EwJpt5xeAtDt9WZMRMJimyv2NK/jJpFoBGNwbnKwI2arQi4e7UC00H4eongkkLWvutQII/zUV3vC7YFnm8bPA2ERwl5pGV9b/B/kk9Pl7Q2i2S3I90wmO39w49uoMRCUw6rkwwxaeuZXXx7b08VpCZl2/jRfWwkR3j5dVGWqKqSAcseRWfhZHXxlcHyPg6ZFksex+uuNjduYBQAd8Ui6ei6By0VsiPB2rIvmArhgplA9E1HSpmy0aWCIaG3HIEpRPfC3e7tiQVPn3nZeZP17Q8LnoaD0l19D5K8NyhJaIYg495hJ6MhQUDqqV8KA10BTNL42SlRfuGRW1ASiE3VTJB42ytJthzoyc+fIBIh8L1PXo+4W1A0G5r036/uRM+KNlCfVMK7Q2w4ndDsurNU5MaQBEAwXnGYVKH9cUu1itP7qPf9ou3F0M6w4RcqGzm/5WdRUKP5BCw2HUfARxCE+v4Qs2VvD+Pbseo5/N6wmTMCwwOU8HBNsegM1gIK6xZqaTnKa35IuoDctFmihnw0c5EbEmi9I+uWAKYX0CH9ssmAx3eQ5LFDhOoUZa1fuIYnr6tkNrOUxedDEn76dhfuQwHAGPSHGsQQ5z+HGN6F0wg0Gt9HlODv3d6L0uj77WFprrv9UeAqfj4PXJ4CRIJTfMJTdap85Eq1UbaVyAEBk0Yo7w/KkZWFNqolQvpZyueDghja8qYpANZPliKFXF95c1T87iB3vl1QXy8GRVS53pY6TWKRR+LhYIM2xUceiGr46WHKHeouKT/iww+mkwNb0si9tIKnYR96/fr1w3BYON5iKZ+ItF1DvSKTSvB9vSBqQAkCrPcQO1XVFc1jray2i+19vygcplGHQjyZgf5aoQ7HRfr7q1IIZoh0NdC740JTnrS7+4AIFtJ8OCQDHTSVrvQ8OCRVlw+XLP+QtAUJO7ozICKURblbex0J2AVJJBwQKKlUKNl6ESHI/8hbUzKocaomVOy/P+cg7xdCU+Oy99wJGe/5ybAoYqTDU9O22m2/LG/yaE6GItImAcA6txUGu0lSY/mM635mntTdZm3ztI4OFl55e1RElGNdVAKfvX8E9I79zaP9klwyEIfifxl8AlmusB9TnogthIVlKrBVgFa24TLxyWMOT4+FAIQC0mXu+VhWzkZ3ntcIHWguZ71D8YYphnEiUqsxWmXALCq9PV20icpGiyTSKcB4n/Kji1LMILcSLP4q6/F3iuRu7Eg0QrQGN3YU3Izw6PTbn6Sz1+cln6oTPGG4fVA14IVF/PXLsExhrbxYRCmntM+1Km0xMVjKD1TGwjmHPMdNo83INNy7Lc30J6nvuzoyax5i5njnT+79BEdweLFKNQyXuin5QZXiyNJiVp6FZRQzV9CoDsPUQTnAvl1BpTLQeCoIa8hU2Vm1XgbhRssXhwDRSWyanw7LNFArO4Mal15IDImzlt9ka03pPhhiOWtLXnxIbmDSoy5XIhIIjVfUyRRrKRT5ui71ifC3SYW45+ApDkwskTGHyYPlgjORPxyX0i6T+6NiymfyaFSoc5qMd6a8Jk9GhRT6pSwF3xfVxTY9Cx56/WabvXBJW+zFBYsECNU9ylaP4rZ62Wv1sYfi9vNxr5/dXqvdXqvdXqudXqttD8VtteU+jnwxLXdtEgJrX7VIAJx9CO756SUPvU/L03WHtsMOqYO1HsNL3ibytbJqcmH4tdsF4JLWdzCYkub5KTSG/aGuLoOmA5lpe3voH8dAp0cdURyWPI+/41FVyB8ImW8hvlHBNA86ue8roS5TVK2G9Qe0jT1eIvUNxQ/8NknFvSWiIHN3WEb/ejYoAucgvJkvxRTPekUuHw1Jk03sy6QSWDVicHDovadHdLOT+eqwrL43XeFBGPq5YMHB5l+1JVM2/ZLd35chghWioBE7VqlkF4GIFgWbROW9NOkSkACSDPYNFZR6Hmlp2gqFFoTS2JL5cAVJWDOXiGHcT5E/XkUMoXvloyWiehl+MC6n1uElF4bJBZVYUJ0aF+3W7CVXVov203Epe9W+r6J8xKGImhpmHyL96riTPClLUd8eFIqpco4BAKnj4dDObnhHCO6BwhYyESf8uu060ZBteKkf8iJ/9tPGvP7Tsi5XDwf9zVjkYFHFohIAUtHPodGP6T/+89cn7UJpnP/rydqP4/uMBD1SKrxD/maVgO42/KGfjYdp+rhBfL5KT2PAONU//LU9/O6v6EAcwtkE44FIxVys3BktryuFxhvy7QtRqUlpivHuqz8E3U1ZyIRwBfBj+fdVOlAIwypOSoEp9r4aKrLYQozJf/vv1vVKd42Rgb0XByF4QfifPzlYEx3CCVr3v/x36h0DQP/4k3e+L46yn7zTMeGf/ukjAIeCNf7pnz4WPqDg29+v0jmZoxLCsz488ubH4GMPwk9X6a9dbPb2jtu/g/C/nShi/qRx9vhsnM1l9Dsc66uRczQbjO4CIx102+p3J+yhAyQz//7HE/aG7h2agCwVC5F+/dd0pObPQqRH//eJ+lkN1G8V0Dvl1KcZbOcb/F0Nzd4V//6G1n4yf1dDs5n/19/T0Nqojoev/8+/p3Hp2pL/x9/ToCDy4+Hr0RPC4JAennRD6NZgI+nq9aqbkn0/Jz2k/DWbUq+vt92U3A7eflPyKk29TemffoLeVMbvZyfsSsE6/DX5c6turb1z6Em3Vm9Hxtb6/Qn7GWT96oQd8nrVTcFu+Xe1y/9NhnayXf5vN7S33uX/JkM72S7/NxnaCXb5v8m43naX/5sMinb5G5abF6MOVF2IkudXfcyyqb30mahveX3GaNEu0oW4McDez4uDfM1h8hgdw+sWGU2X8wJgQy9YDAg8OGdRkviS9Zus1jaNIinsUWPGG6BeQHRAfRSEZ9sQF7zRD9VPiHCSG+qHljb9zw7L8C6vo99LJ3xhPKPK/XIZ5F0AE/99PwvjHBjiqTf5PusJAZgqhDI5Ye+d3JJABFvwltu//fuByyZfHUv5j0dsktQZL2UQXnOlIFQZDfqbPk7E8wWXdH9nLIGH+EubhzlZNPq0f1WQLSp5Ft6b52Egf5WH0QqudVAwyv+D0D8eMvj1rvTJM1F7hKG00oZLprQzBML3svpUxGrtUbG2JkarUUmQfju4/HWXahjhb/Z5RobiNy73oe9Jr4Ksc0n9TFRMB6wFNP2olom6C4xyVj+eGOzS8Ieyo6Kpm0VqOmQHZcE7uGPLvl4qm2Qz1nwCKwR4wScJ+Mk4pspOd7Y/Wdo8qoqSQtMmKdkNESf3V0sbUpa2TKFocqIAl8CB/GKoZdZIwWZpkmXC8k8PjhFqVUVltiAnYv9KO2/TEAvdBuGz5W1QkhEU9BJpBnW1t2An7F1o8DGAigBgyyKeCfMxfLxaA6pQW81U9aTng61SvuCsmQN2FT05kBm8fUO8o++uIP0WN243msA9zaokXuUWjHcy+PVABSOYmMbTeblMtiOyRRKLgk2hUt3wfatm0L8hj9cKRhptr/ZUUdA/158ONaK9xb6XYeFYTJNI7qy0Gs0qnie1NUfDQdk0+3hhSg6+1GlyCNH0yY9Wx4PzEgrYk7DxWgfvr5PuyB8NCgtRYsmBaLXlyJI3PkP0j8ggHLxl3XB38Vbj2t1a7Yna8inP8asafEZRAQAU24uV7rkT9k/LoZsY+rKG3rP/k7rVl4ZYEsyyFHlMCsxcpFBDRLcV4oZJxk28KNsS7BaTisKWHDbNOhHyjs3U0SdGeM9dv0SOEc4SleWBTqBEkBrDpl+ii5TZ8At06lGfBcEzVr+k8hpKzKSZksZFsdJQEpRKpq+11IiXncKz0ZLNMnkw28VFxcrZAV9Q5bibLUkdligeRt3xJYcLd7LT0twYn+MBThDSUFUMEOmMoJde6JFp8BoFh2quMSqVd8VlZVVRqrttg8bNXw4rLYp9SGsCte+6zcKHvVBJ4xseHt5PENLzU28II7Pi6yape2nY4LzLswjwLNcNgho8/ThHf+NBDQRpAdfTGgKCEExJ1tO6ywTBD/BuT5ItCgCsSVlVUyDbEhEY6b1xEZy2a32Zsp3NNtmaVDf7AjgVaXC3Brl40ft9NqnzWAUBRUR8tS+EnC8MurVcHswTieVSj5dIBOGTpX101O+GZNsAZ1nuj1zUFNMK2tLeOtbnow34sgvzIHy8rAdXE/CItnGBpvRXg9K84vtC1COD0yJB+NHyXqA8DKTgIyrm0+Xy7r49KtzlWQbh7pA4Hp9kCShkyDWv8c1goyQrtPjgg+iEtBKzpKeOMTiXTHHz9n491EB9nIODRH4Qhkvad7R/GZJUei3EkArI0xm8pC0YhNur9ujq+is0MZ/R4PSiAkSNdZuDa1Fbq3X4/khiZC1q+1jh+2xlzVv5ekg6Lmq5kxe1GBlfK6OP7eP9dPTB5SbeKrI8Wel2IP22qBxTxeBCYUibt/8HQz4Tqcf0h7xVTH+t4LrRZ1rMoF/asXHC7+3Rxt+/FHFWuRT2QJfqqoUrzFcyI1/qMYrpXRU9KlHVbfElAQBB//1sUCQq8siIs0b79oeEJI04ivJixo/YRLTwj1mZvpPxJD+bcVRCY8ngDGT8hIX6cvcT0wBFPqvn12wiVNnS6AA2B/I7iup+xjWSHVxZZ4hRdjzAH4O57aFfSNLxUGixe34xDLrXMg/8Mlo5U1J3/VLmmF75RahYFiaEaWspnbgoL0I1fjHcWIW5QjZvHh3pBh8NN6CLOPJPhuUBihzPf1r2s2FZcViKCPLz1ag8twJzIit3zCmifuopon7aU8QgOlPE4NAUWdfELtOE4bXOZRwATyN9bfM3XPyK8bu7+oZDNS7vNqDrw3DjCqcuKxRSIJN1nBQOB2ouqrOfw4HR+TlUtBI5G8TBATjdXXVY2F8xm7lNjOu4TfBCnibGAG4DC05uvi/xgC9G+Vm5sznGhwve9wkgW52Y4SrLhbJy58FSofHrYWLqwZLraaHR62khuN7dcSm42jKRrNy5t0QErnRzUAYuMsLNyp1bw1zour28qDrXmp6OCi15SAbn3xIZnHBK5polAyPQkhsGx/kYXJY76Q0WXmsxlW4T36QHFq4FVCFE9zzGh24ejPC7RefxEiljEVrWIS1KVwFDjcl50QC8MoD+UDnS0Acr7oKKIwbWziAKuU+ccLPOOW0+wj8pBsLtBfGxXnUCoPLc0SoVINPUkJzNM2Yd/i55JE63BW9/QZUroLDhPait1TqU2UHZAu92TvfHjoyZr+qIOt01M093T0ZkOsRvwuW/v4rso57Q3t5xjxaE/c729nok5w6k74E4HVUH+4u+UNgX6g0LiEH40CfZp92xSWXKs2K7TTYEwPQbYxJf2Mz50aRK4jjn3ciPl0gE4fOlfbTg2WBQfB8xFB6Gz1ZvFuc8XFnakZyUxoyEyAhA95okMPcf9SR7tzspRW+KoFyPdAkpVKAEy0pPi+pmV91YQy/KVg2Vd/tcl3KnL4I2cH6ICPAyCK/3JaAadCUaKQyeBvxui1QaPauU04MEsT/adNteaWYVtlRyPSv8zCB8MNayS+cdk2rv4eOelNKlfWNQrCC8N9yqA8IelmmvftOVqaTRg5fbtv3M5bagFmUqPKM32UH4aLx1O4Ylcu1o7rtyhE9kP5IxobanoCdUSDZpwMTJYI0afsGdoCF1yyO1lN0OZsdlFw34JKTn+SpOEN4dbNNedVikvXJvYJnkY+NG9vDkyESmFjW27xm8yfZMDrv18OSw5drRfDEil8gDPj4gkAjCx0v7aIe1XLQdWW8R6kRHB+VZhMyWw3PUlGpH8awv5V0EgRyEt/3S7TUfunxSN0H5kiuLDS9RKR9bolLz6fam+T6T82Rajz0gVZDeXoVGpdrL9fYcALpm5p4zKNH20Xu4BJbd9jDAH36VsVzsel4lkD2vkqSHr0b89mof+fhbA5fbCsLNAfn2gkMC7RV7bwLQTKT7lHoLPWj9LEMc6FboSU8o9a2rUSqDsLfcoezw9aj2uoDqlysJDe8+KJQXdEIbm5IoiAGzYx8ZSu0vnNV8XGx4VddiA4uoyfas6nbr4VXdlhv+upWcIfKpX2R8vGq4A8/EHe242PAWpMRiyZeMRkt4tqBeH8NbUE90eAvqREcH5dmCzJZLJqqSakex5ZcaHEEQ9tZS3WJ4tdUSw6sJSuCSsvTDjVIx9avbbWSJCsRt1faPBmXwNE0AnLhDUfbE89XlMbB7WBz9OtYzcyQUnGc1IgHabcLl5ohEAfc53AXWYZJB+GxQArxPhr4t8ugtpGMRBeGTUWlbzxx+vo4sfX/hiuLW+cwjSXvveG/dtrZMEicjqk2W6u9IxgKwyIumfjoogh2R+492myAMlgvTJjH81E3B3jI2KKtEh1+QIdotkMMPyha3lq5ByZX6w/cUhBsgiXBTNfpvawVJKetqvWXFoob8Ui1xtWVQaQpNx+fTfea/ZBMsF2RSEPMySw4d0Zc90Zet6LPlotEcCsilIP2ok854NuGWNFFSfiSqYKncNgkaHabTzBKD31iAFJS0JXI0jafT/A8oh2YusE7ZTvWOt8SpbgviYYJXAAvclAAOKmHhrAryG15puQm4CKOmArzgay11zqVVbO9yy9EzIcln5wnvdlFEfIJF/M6ZBClKSwAU1jWTkEimis/cNsloeG9KBOVTWS9XXT5WfhClly4Kv/ykkJdM+jTJmGymfZooe7TKQysr0aeB3d6hZUlsPZdJIdeN38oWygTUTvEzIKHMYlBRFTBc4mdtcAj+2DB+U5ADwlc9GZKEcIsyEQAXXijvyVWPLOzK1y36DCvDUYleXDYMHhZsAXsulPWyuzMKwVl0XZ3z/2fvrcMkN65Gb7HWvMw7vLvTs+Dp7uGsN44hiWM7ccCxHcdWNC11j3a6pV5JPeAkDjMzM7PDTA4zx2FmZk78PVUlKEmnSm2/ud/7x72ZJ97WOb8qFZ4CFWBtFOOeGx3hn36zuGukwctN83Ul1pTWFRo8kzzGodoUXZC97jaWfI8KZ1KTWt2e4TXRhx0rCmc6H5qcE57V2CfJBjf06bKgwQf+Ic3evCbei4BucdxBK+MT11t2uDmS492S5KvLxozIDJfiONlhfDP43lSC1pW7xLShaAV2uDtV2p1FG+/VCgg3AKqIY8ftbkvVZCE0bl4GU2l0YFlApkjQGvLAPrmDoR8uyKOZFZSceHZle5ForNiN2MPoFvS1sEGWv2wvyl2zYwM4vndpd1YernedhtnGTvbAKuyMocNN486cDh0pihdN74IU2LthSBN/sMaegm6xZqCoQZtXWh0Tx2KQrcav5uhZL15mBnq5NNDLRLM3p7E7XVQBiMdjDGXGb5YPYBYgJfZ6CFBkvIVcYsWWrAKXtX1ZWdd20cdbOyDavSwtKgtMpziYFZaWXhAWj1f5JA7KWBmFliSUeoXTgRlwqDBSWnxw/SRDjb61gYrNWTlQoDuO7yGDaazUSFwH2PoS51B9oPVQuUv0+N17GEr0YpZD/FaWEqpGHcdl2A2i4VVBQkAZmd46jMvnAFOLXzzBVBdK6IFSFCfd/lIMldFyz6BiSGFYnWsj8HqctbCxGxBHln0npEI2eIChiGwoy93yHkgR2bAdDN1eQJ7UFCggVOGF3heroUDigrILULR827bWx0ENuacKn5TQNdHN324ApWkEQrlg+flySfY7oYYfH1hRCBK+QgjVL1iDTyIANVDzH6lwIcmpSLyJqxFQlSmBsGus2pNXkWTDtaDgjOhQ/BgqvJkPVkE9lViHQzLM0JFLRtCgIFc07E43XIdSmyigJpZo6DqcK96Wvw723JAcvQkQozgDvuDXA3KoYcdyHM+8wgkiY573CSnwJjlADvVbsKJgDoc4EH7tIAdAycfzAIwpAqC2gChwGuQ6Nzg2ObqxZDouCeFOSIOyahuoAHGgjSGK9JJPMAA4aPshTSGlh3kUjsgQj0BpzfUCp+kIl8CJC8YDO94BaixrNyiHcgpq7VJjCfZSKTXKM7YWyCJKizOCrcYBG2SrcWS25vRYmKnveKBGd1fQHlsjsPGucQMf8U4P77q+3TC6XttprJPpjKj/GM3143mMqDh2bb8Zz5OSdIqG1GgwjdZ0jVeiWR231zHpmtfx7Pj7Zk4wUgDiiZUWuWAXrxADEfTfezJ0+P5DaB4EkeV33F7B8NayV5xGYS6yf3/3x7Blt421FdtHk594ooPMMJJVbgdAqiAbgjA8mR39MwYB0fQ2eh2GzqahVXMlnSez7AYdOLI0MOibjyKzUMJztBN8p3RUj/LRfKSP0PgKmseNL0asAVHuk44iPMOlmbppnrNoqURy73lisG+RK3oNAtsVnazcmEcgSdsltDeXR+ezYICGo+M8ogkzKzDcvTz1/qyyvezYQLaVUFBlS6iCbAjCmJUtBvKRziR8OE2WMPtA2MvBKPiHWSAkPsiA88EcznJA+DhEFLCBApF5Hsmr8YuPhX7PPh49jMJI02wHCTOUZ/JRyRQCfIAWFJ0SCioqCVWQjUMY+cjYCbvUzyEmCJWpGOCWKXTDQ19FhQYh8SgDpoN3kMHkQ1gBOcjS9kFGkTnKJEH5GAun4wOnJGC78hFfbNvtDhCdUi6KzASDA6QjMEpH4wCM8COBDjeoGVDdKOXASKQcIB2BUXYkEiQfiSzW8drtTh2IQxkWRWEcxorCYRCkw78fJPLBP5SlzHY7XHLcZbBJ6IuNIlLjsAzNYbaTomk+UgZnjPQEm84nyMYMSrZVUQK0OMD10Ml01dJYZ1mG5gDbCZ2bE2yMX61stD0arTUrqVYAB1arlAOkIzDKrlYJko9ExlziFSlADEAoLxoEIDpAo4Ce237glSxojwjUfhxlkqC8EIMYZ7YKNFSaaDP9JNoMELhCos2UJNpMMTTbMgyaAkAj+axL14Kq0QjA5CQDRYQOH+ADt1/m202zEZb1uBKqIBuCMGZHKgbyQcqkDcpjIPu4DJR+EZOTDBQRZvoRNdfMxFumykajFJffN4Gme6hx5S1zlgwsJ1jOilIGCvWTC8lBoZBdKyDR6tW2fbgPjIwym013qm8YLxYkJ14cuUWOeImQWQPbX9DxqkaUJZU+4KVGtGqwH7LhdRYd12YkM9dg43PGVtodDxp9H2WSoHyMhTMNNg1xO48RWNZ5BDCo85hiReEwCDI7jwmRD36h8NTxCW2QtBg2NzpQkTdGpEFIfIgBQ3W4YDhpFupVFplc9Zzun6bqZ7HAcV0VzGyEl5rjOlzei3kGl/aOZxesXr1Q0oslEyznxVoDvLFYRHKJgmazFz3XLuZkyZwEZqZz6QHNc4Egs2BOwymHxMWCOZ2mXWnBnC6kc7FgZpjSgsmkuQWT66pY4Kb7tHHTUKIB+dDHnFUGjP6pcQBG1IHsYjopluWyWQHcW+/PYtcgi10DLHat1GLXIItdyLQaXCtKOZY1qYF1ogZZk1q5Nan1Z02g6TkQyosGAYg5zGEMujJ5SJ1Tyrc0IAhZGhqExKMMmDlNSjH5qGSyqAtOKPKQKPhDRSQr2FcA6MAOF7Tc/MeHsZQNcwsQVEhiKC8aBCBmIYn03MEmZqBJwhIKGpImVEEGYsWJtINsLDOFVkwqoNUtRgCaIymhmNGsAdGsDUEYnT11DhAPy5CQ3nNU69cNaUOQk2L6ALYiG/MlBza+JRSYPjFVkI3mRFC54zLQnELE5CRjRaRY3vKRg0tbPkRQMeIyjFDni9CSUxsoIsyZEKLmDoO6ptvqkT0YQJj7QaOgT7JRWLGf6YCOT4VJ5aOVbYZMy2rb+PYLUDzKgNnNUMpw7bvn45UDfPtegCD7HkN50SAAMe17pOf297yu7cKf48owqFuYYkXhMAgyu4UJwZ0uQZTpGLBt6oOEJlYyJCgfY+HMiRUa4heh/NQPWIQY80MglBcNAhC7CMFzQQWmzOTlGcjkRUxOMlBEmCaPqEvTFzLQfIiVvrVieGv7AajYtBxgUuy2JcK4y7Ncr+M0jEXbDxmVok8amhAv0ExdBVaV1VGYhOpohgTlGQPu2h4rAKUcNHqkOEA6AqPM0WOK5LM2m/x2xwt9zzWWgOTP6CZ5zqB5isM8B1wDnMKlmQuSYObSJCgfY+F0GjP9zHSOm82oUzzVF452nbh4bXC84R92xJ3MiMHy4ljk4OKYcIB0BEY5xTFG+JEw3UXbaZXOyEAcGImUA6QjMMqORILkI7EpgxnBqrk1KyFh25UVkjk1vH15C6DZnpXhc8y8XgiJHbe7uSjelhU12gFyn6lA6b0qZVUNJqGqliFB+RgLZ/Z1aIi7hCd7zyZ/CQ+LhRaz5FiG5gDbCR21CTbGnZLqdKHWhodAU1IEyQr2FQDmlBTWcoeA6L++y2oc+0GhIWAWhRX7mQ6YMyIQFZ/nQ5ZO45XVFaYbbtepgy7aNttGG62fnirrOrFpqOtUoJm6cZ4zOmUO80B+nhMYDsbUQUAITcSUcpCRpzhAWodRsnoCn1qIjgGlH/FxWOVukpUak/3Q9NMI7IDZ7qQIt9uUXktTastBErTlNAnKx1g4HZ3JEog68akTmEbTHGc56C8FSlc2wyQ3BWqMFKjBKcBbw0ZDfUVostpvhCah9WRghCarcIQmq2CEJqt9RGgSWGd2IA82uqVzPilWFB4EZH3YkiIH25KEA6QjMMqpuDHCnT9A18WVzh8UIGj+IIbyooMABI3ZBlkcNHUQ6blbVzp2aKJLstDZXMbiuuG4lr22s0As22j7rL8PVMTOthe0+KajbJtvt9s9qHgdAKmC7CiEJcmTmW1BgrP5eGbaBUnAUJR8k8pi7G9SMccvaiY+T7mkqOUhsKhFUF40CEDsIkT0JUHuLJrlQc5BcJAJlBcNAhBzDpmcuFf21apIgeUwpgqyrFEyUS+UtVGtHxTsT2dQWLGf6YD5SSVDcdcKoBvs+topRYOQeJQBM7+9UAx3vIWChaDoxjhj1XZaS9m1zjFCH343DAEkiu02vjPoGEhgWXT/B3kVKDzAdkzLwJjgY07QRT4LC0tmUIaYlgXGhRy2Yna76JA1MDjkihEMRHcYjrIwnPZoOzw+ApzN4KPNi+omWlQfuR+HAMtz8Xl0Id5LD9TVGGws2Y3lOKktpxNUGkumf/z48eMHDg8ZF93FuP1Fl1w4cTj7wR2X1spg3r/0JnF8Islutj5TJ8lv4DMnpagwHXDNKP5dZiFjKC/aD0Al3x8yFPv7Q4RxFwrg39DguYSCTG5CFWQgVtI2ZzF22xxzXHvYbpuWyfjCUg5ChpMGIXE+J2KYaTgppqywWaVtdgFilEirWCKtbJsdQcxuRqTnFzKn44Slq1GKFFjIYqogG4IwOtxjEMD9VIrufi77VJpnoE+lEZOTjBWRkjUtNJSpEcDbuGPFdrPDWJ1ZhkFDyhQrCvPhh5dd5lO0dE1OnmGkeq0YpskiQsYQ6VES2efpcgfApOY8nYl0ag01PDcIDxymDqKPJpKJopIvFuD36IVM9t8i/ws50leJKltx1jZbPReySyUUWMdjqiAbgjB2HY8A7nexZa/tLPoONAtUykHTHBQHSEdglDnNkSLcKeJlp+NE683LhjEMFBrGZFFYsZ/pgDmMyVBcM4XJ5dI97wAGmakUKwpHQTCKgm8HaDfZMI8pfi7LE9Gp8u1oHejhchbfbI0XaPTh8bIVeVzMkeWyyd3lmrHk+c514CfuPkhoLjRDgvIxFk4nabUEim4A8Hrorm58Zfw4ywX34+YJxzXJepgVqLz1xUIfN3MsQ8PyHmoH+2JLglJjBiU75X6inwmiAgR1Nk8UJ4hOFCaIThQniEYBPbf3cMJ0oB0SXAbqPURMTjJQROjQAj5wO/MIKZ2eLkBw+hIoLxoEIE76Yj13KEWuqW6D3a/DLBASjzJg5giJYriJurRurEBDWj4EJWoM5UWDAMRswrP6ZNNx17cP9oN5QdgPh/qcR8q4dDWUFZiVvulRgCzNAcgyglBeVIht3NLxPqjQHFSyIz23FVzques90zXAfdJ9kFArmCFB+RgLp+MyAUHwAK4fFOrsZVFYsZ/pgNnZy1D55J+CSHxJDhwErMqWD79jhLnd3lDvHeKg3jvFAdIRGGV+uy0gJAGC0GwsH4DBfBJlRjzokp74qrP48osdeSA6+K4gt+ziMXkt31ux+9imCXFQ8lEcIB2BUablTBFui9/yPWiJHZeBWvyIyUkGigizxSfqfGDns4jpOuiql+hC7+vZytximbxL/nonNg2tdyrQ7EDxnNEJU+sDzO0tn+vbSW5Z72GeQ+7AOYWhbm4/KGRLsyis2M90QKfi2WUUb3s+m8+lX4Xpqq/E66N1yaKwAiy/8W3P/ZX2Is0r7QnN1IGlPXZG5xNYAmOQO8cRwWWTFylWFA6DINusdkPX9tbKZlUADAxYghWFwyDI/P6fEHxz3w1Lp34jJicZKCJsU47V/JxDZwAHkN0ow8B0TLCisAKCUKd4mE2CKR4T3CFfq92ZNhhn6syywJIzuG6hu+QQrsNMd4D4EAOGUm6Ux0b/HOEzuYm4vujkVKuJfmgybuRGKzd27CsQ6A7S+J5LkObOoLfanSnGYKSUAzuRKQdIGSiUpSMcFDSMCcI3QO0ONL3AZUAjRZicZKCIsI0UVnMPR0RLPPBti0ez0k68tgHd5geEdI6Ps5UVrkN2bcqT+VidzaGhKbS+eegof4DnaCf4Tuk485I1fw4EPhsP3VaFaiYv+8omxDBeNiFWgKA5mxjKiwYBiDkXE+nLM3jKMIPACULThcb3ffPMDM7yHG0xgzNO6ZgW8ymDcr+hErzsG2qRgr6hJlRBVoewKApd3zuBLnlOih1phbObnHJu8D+THAAozeOVYrECLFkxRnXoC1UZBna3EqwonALBkiQa5jkqzosVCLMXBI6JrubtLld5JJicXL/Ndtjroo6D5TTCPsiG5/t2IyyWujrwFQ2g+iqbdSjpQQy/hr1QLIuxF4rFXHkESpcgFylmNKGmo1asTrlmohhwwMZndsbitZoZZ822A9pNPgRZ/RjKiwYBiGn1Iz13oNE02w20fgZajFEOQh+haBASjzJg5kcoiuEWJMKVFaQiBRWkhCrIhiCMWZBigPsVxF4z0arePo5ogUnoK0iGBOVjLJw5t05D3JwgYFlOFCkoJxKqIBuCMGZOxAC3wSMQ1B8ow6AGL8WKwoMgWLS7sIeA4QWSFmjes6/t+V4/Z35AHDSKpDhAOgKjzKFhiuQjkZnntn3XsaeMaaCoU5oDbCfMj35ZjJ+SEVqakgAHpmTKAdIRGGWnZILwa63ZatulBxkWKbDWxlRBBmLFcj9diqWdMnxohO02xisH2a7YvZSY4w7oLN82oZ14IJQXDQIQs9WO9NxW2/LCoO564ArvchBqtWkQEo8yYGarTTH8hPXCAOp88CEw9SMoLxoEIHbqEz13G6jVbJvBkhHYbXxjI7lNen+RKKtKRQqqSglVkB2HsGgqDjoWISedYDu3gmRpCv59mI9WMw9g6EtqOIiV1vCsK3YNjzmuDbdsuxvYNvRdu5SDbDjFAdIRGGXa8BThG4YIg7owh1kgJD7EgEvn8fNs9A/Xv9wEOiugebbjrNlBvx6j1Vq1ftj0Qy3a6wZmPdCxqoBc6UEHMAl16DMkKB9j4eX5UM2lge2ii8gn+2KT9JqqwgWhurBA/ENrYhjFFTC5YDrVS8/CgEleitZrcIrWa2BBrDNOGwCTv847OIOGuCulY7BGTmzlLk9msdDy5BzL0IBhhvKhHAR7HhQIieHY3QJLxFvgSjH5HMh2G8hKLnyLY2jg24Bz72o4ZR+o8gz0gSpicpKBIsL8QEXU3K9p1qIPLUXgMmBgCZOTDBQRdmCxOh/YzOFvDd82w+wOg4bX6ZjoAvmyMgiCUBmkQUg8yoCZ5YpiuHat4S3Zvs3YGNgHCdm1DAnKD7PwUrtWgCG7RkPc6ZYILJtuATBouiXFisJhEKRjsB8kuAPmhtdaAQ+GLqGgXn5CFWRDEMac5ooBbi8R2bJgyW5Dq67LQbgCpSAkHmXAnAqUMFxb1mg73TJblmcgWxYxOQmA8EtE227mFpgi0XhltEiVhQczOcmhIsK8yn2ynM0+9+F5siZooMgyv01m1PTiSerKnkrfDoCUzHb1bdMqez1+MRpXNsxeYLbHK0DS8w3Xkhm2ys+EBzDQcCVYUTgMgmzDFRP8+r9kduy2DX4sKQfB+k+BkHiUAbPrf8pwp48W254H5QMfgqaPYigvGgQg5vRRpOcaiUUndG1o0r2EgpqNhCrIhiCM2WzEANfWMr4VcBnIthW/ESDJQBFhJ3NuMhYOCThhOwIwOclAEWH2YBfBydq9GcQM4j4bXhZWYSijSBhLpmV2TN/KnSBUJFlvCUIzDFhuo2TIBcJpO26rj14oTEK90AwJysdYOLNjSUPcZfIpCH3P6AeFVtRnUVhxhOkA6lvv59PQKpYM1WciQD3sflB+ItRYiVBjRKvWV7SAXvfBHNlY6pmlR75THCAdgVHm/GuKcG27ybrc/ABIFWRDEMa02ibjgvNckBqh0yhrbooU1NwkVEE2BGGcgBOA27SbfsMuPd2xAEFNewzlRYMAxGxzIj23X2h2bT/sQZvhyzCoX5hiReEwCDL7hQnBT/FmH1frFCAwxZsFw49FgwDETvEmaOn3UMzCQnTCXdBtO+F4ZVdGh4X4YbyyM6vB3+6MxV4zzDqhTz7ckdG4Hjrw0GuMV7IBcI1Wt0cGKEH+NWj/7IrTsIPsCBAfVxifzeet2L7vWIjZnWHIzJfRsTuev571lyQJTo2sAqcSOT4w9qxjdhcWem7H7BpN32x1bDe5dAGr0t8ZMfr4Qf4be9T2GunHLPQwXtmUUaGDFLdmJC3fWzVCL04wHBPcPekF8RXgW7I6+6Thd3YWZF0vMDrm2u0yCt9u9HwfndfS8Fyyl/d6PjBeOVDmg4uxw2UYWuTfWzTxntGJfuDAuc5OP1/yUaPdl5d+YFzXp5d+n166BirEfaFooN3H281ut72eHmaZ4xYWUGmwjVUf70i0Q7MvEJ06OtoHyAhdzKBzQbFfI+XcABPBRZYdmrj0Om45Y64N8Zhl207OYAABy1nhhrPBd25aFlPf9e2u6duMorGwEEkWfdtctrxVd7xylIXCcmbq4PNZe13LTFehwEyz16bmcmAGV9l9LKTRtk3/woyW7Cg1nGDVZJkZABmvnFfuSxkxzQI4ARiv1NiuWJr9bCdU4pdRJPnHSiicAQdByFrLm+FzS7kSYKLUA2TKHGstb6EgNDJlcAyttYzlGS2BkEkpYyzfy9XXDIPtzjBXjwzGEJewnBX+KxolHpiWdYgJFI3CJJNlKNipTRXNEoiUTHZypwVzgMlg0wAlFZ2jUFJlAsoBSCDhN6QB3JzR40Btz4pM1wiWnGa4OyNe7DWDhQW7022bDXsvoMJ9W/Q77tOZa3Hn9QwiaXstI7Dj/jJ6apjt9qLZWEYP97DjrtzyitGw2+0Af4vYnpf5NuoJzadis7Fk43tS8tYVUibmHXZJrMYMD+H4zXNG9/dGeWBkJCpFBoxYkuNZkrIkA2w9KnK7QTW2DbDPaX+ErzfX9rL0yKzsYSktZ4UZpgbbmWlZByBd0YSAWFEGxo6qimw9qYlguqcVESgEpJQkFTAZjWYZXGdvk9OA3QtQO16Z47rlKA8BOvhlyen0ObZoySAgY8kgACfgGKjPpeDBHIQWmBq+ucroKHC4EuBIqQe0BZjsi+769gpZKxTkkx524BrLK0kHjsMmC2PRaQ2G74X1W+jEsdaCvLkruolMWb6kYzCX/vkymWU4yrydzDrEo2E8XVCs9kXSN1f7wdqO1Q+21DDHK8tcrOF1uka3bboTQ0FoLSys4PXgx1I3R48ePb6wYBhRu2vgxjJoe6tG1wyXcmC+QmTfFWXGeTDU6ZaldkqAhSXrRaGA5Ws7WSBOtVL52k4DqJnK2/1Ij9upfUwdamjyrVCqtZwVtrcNjkPTsiDDAjU2UC0BkxhOIY7BpADYYFIANjv5VpXocXNyfkFlljQpeQIyoXk/SoBiuTIZTUysgQypmRmrFhTFSmIW26YSKDMcYEHE0jOZXDuV789avP6sRfdn851Vi9NZpXRFuwmnM1Ss+P1Li9+/tNj9S6ukf2mV9C8tXv/S4vQvLXb/0uL0L63++pcW0L+0iv1Lq6R/aZX0L61s/7JoFKNS12mb4xWmFjduQNEu7ZtaSd80H7BcU5+vXGAHKW9eWf2ifDEGW6AV1AJN9Eeink1fni737SnpLuWzM98w58x7cWKbC6BJ6D0cYBDW4Qls5DlPj/zezdaPFVRtj3wBo37yIK9r+2bo+edcy4T+K/2j7QXvsRUqJltqgdg6M763MatDlmcHpLCcFfD9DRg3LWskLy9amoE8kn0u+LC8Qr4zpr+uYSH/lRQvJB9l3mAdMW2F0sbsxNDfxgplmG+xFhaajmuhsIfb8hpsxwoVLr64oGsTL6NBmeNFlXDJC9LWOiMdr+RZdNlZkUXS5Ma8HEt+k4++u9nESE5l9Tqd9azrPRykkuhw3c9HKREmTUiWtJyggZee5fzJRzcRJtmWJfFPEtqdLH3OIZlspIIQ5SopV+iCOfyNO0q6+ISpBXTffAflaRKSROXiqo6O7EH7R5Nv85Q+OspnT0GBN5x6vRDyFOvwrlS3wXTruN0kEROdExiResVonURXyfqOuYgScUeRxO38FUSOds91TN8IfafVsn1UV9HB/FAlR2hJJT8+XrlrP/4aRs9Fqe+Ybec62yLLH5ClNxped/0YZrAMvW8w66Nv4yObGqZrOchcxMtDYn3X9AN80yT6F33i79luIx4jwZDfa8dWBwR2ZHW43KD1Czk3prViug2bHAm+M6drNOxuaDSW4jqGrZPZ6bZti/RYDCdc2pjqMoszkCA5IS5AkRoAFckZVltSdSNcM7xwyfa3UTK8Oh2vVj+VSJu+bUdmsOn5HTOMF5EES2bXjqoG2Z2O17gnt0lGvtKWqFevbaaloY1Md0ZE7PJ2WoSraq9eSwo+JUaTRz03MJt2xkm6kiTKITtomF3bWF1yQjvomo14P1I6EYF7J3FzmBObLJjV7CGyr2YvBqP8tWyc48518feSqMtnWHbT7LVDI7OgOOkPkp4V6gQE5gox7WwEr6ICkcaSHa3wQeYR1bJhEEFV0bDX0PquYF+O6Ppeww6CqFeed+/1QtS99e3Aa6/Yhu+t5t2nhO2v5FIhzp3oMa/E5Tda0xaPRxKlvRb6ZiPMHOa2J8c0zG7Y83GwgjNjXadDLfJCex0M3Ps2QpyvoR2V3rj3jhZxRQuScAoYKPdQHbo6J1xYwFkNlR7E9FV6YnCE9htnnL+wQHIgTskhDoIXoA1wANOy9kFqstgtsE8OsrX2yZ7ZHob0KL+8HvoUHiRjmSLhGr3ATgeYEEDKTJBYEpRWtmthc5wToZyIbFZ0s0LmhvW9WVXb883UqO9gKCE5es+hrBxfw4uVaCMQWpaPd/xnBqVFtkDuRxc/ImPtGmbPcjyj63RttKMJ1XEbHQDodbrhSAnVQpuJ2Ui8oesAh8FNFFrf17GHyrBhNoDKZ4gmPZkEmgMwnMAgk1c1NsjSjLGdRGWw2ws5QWy2ewGeYGATaFhA8me8sj0ZsOO2EPcjg9DffiLwXANZUHQftRd3AIgYF2Q7SK43C46ecNwT5sICfjJ6rmU3HRf1CBYWri/IxitDbBwFYrwyzgbQu5H5Ch0U+jE2iC7xXjHbthvuzUBhD5WE5FV7skoPRR5fY213/a15HYr9jqww+bE9Lyf+786Lk9mAc5iq4XN25lWu52LbVHgLqnLLNiAO0JahvDibenvyajOIoj9e2QXoUJqMV0YBjedbtm9bhreIerXjlUL4zQDdlQUkhxkYzbZnhvAbFz2vDTsyfd9cH6/sz6iivAvzz2MMquc6J3s2OXajso8BkXxkvSibosMMKi2MIwyCSvohNkJygAOQBMvWH5IpVGWMBflYpWD2eYxBcZIvhUjysbxwAuycDPIGGBBaDL5sr7MCy8uClEqzYBeDMHOZQ2nSzJlgI/lKMMRGSTZl6x+q4mmCjYK6bFQHQSaN6BCopyJzIAMsr5p+iyom0XM+LAmWyf69MEMiMwQr05BmMw3tMMj7vxsiiO8jkIqXUhFCJUS2eDZ7boNKB/KYL1kxRD+NgEQmHntABGqLYh2x8IOgLk3BAVCfmIxsTmOLS8eQPOdzOsEyjwyGUxoShkRzAFbGTdwYrObV8gSisnSQSZDKl30NklEJQh7zRSuGOBkaI1AljnXZmAyBDBWRrCe43StJ74QhoRiEld1esISHGowXZEzzEMyk5W8nDJi5qp0qWDlFEySnGO9OOwGHI6Dnmj4e6/t2QM5juD4vSktfEV5xAiccZKqz/YCi3l6zG73QxqOvuLqRzmbbCW3fbC8sXJ95TrMth2XNWk5Jvybui4Z4lSouUbjqnxHLPdRjjZus0A7CbOrkJGkjXkBx2gywtNk2vqCmgzyZQH4xt4rCNPchBzhUwxyABOwgh6DDdkXERT2qrumH2RmPVF8y45EFx7P+psUhK0iTPw9mrUleS0dhJMuc8BzXcD0yJWCjrW22tTuDLCz0ul3bb5gBNViJVaETtm2i2ppToX+72wCh28r70vZW4xfsyakaZtcJ8ez5eGVHojNDHNRcqaPkdIzjsWfQxXOsmfJUkKXGBsDx24bYepILsbEJ2k7Dzr0uJ0qNTRHOGJuiOmtsinrI2ATRYsAoka7PPFNpn8VwQPbCuqwhyinBTCDbRrOpkpdRmVDEs5lQ1JMgHWADdLDiGuP33NDpoBkUE31CMKgKHuzIM5EHsaHy7XDJ91Zdw15D3z5InIrC8coRjoOiLC5GvukEtpX1PCcar1SYcF4STwR0fQ9N2cTptTcvplNpc6y0zWUj8Hp+w45fSL4f0S/MS0YyZPq9iioqNRDpoA8+PpV1hum3eoin5nVyTsyG76X+HgahttdyGmbb8PxMkTsAwlhIV5KDHIz2bhjknGbqF5wwTc9PkT0gYvqtIDXFOZ27Pl7ZDqnS+h2LnSB9096izrHQqaRNx/aHckr0saQRUsA2ENhJS8nEYTSpGCmiEW/a3GUFaXOXB7PNXV4LNXcRAzV3t4mQnJgu00zVvsSt1y3axdjoFUrywsL1BVlq9AA8Y/QAfdboAQCdJnF3L1dbUKCykjQDCmim4S1os929gpoOTFyf2vZa1o7kBNtpLvra5Fxnb8uK8T/jlZzUCYxVz7d2FqSOG9ot2z8vUizb64hL7Ew2xzja1ABzvcCJNtEPSRJwsh+UTsw4Q6J45SvMGEtN+5FAzUzRoB9T45OFcPz2gCoSgGFQR789LuOpbUHvTh7SSWAayI5iaQ3t9ZZITxZ5umgBYnsrJTPD0HcWe6EdpwDQFyBbnQNnxY4NJp5IQRY5Gi83fS+123klGgRsjHXIHqLRVwLTlj/X7crpst2unJKOc+Iez24wxo85Je0+bkGbTju0M2+5Pi9K+7JFONOXLaqzfdmiHupIRlSmhhZkaaEA8IxNBfRZmwoAdLDighSvbMBTEXFxR5/kHRfPxOfiXOEQgdNyUX6sLplUvwMiofRBn/qRqaCzrCBLB88ATkIY934bZjcwQjw2xi/LyMmZL/jUtLNoectOOgHoGZntZFJnPFG029lQZgRp1zQP4vzbx1Bmewh5LdQaYiZTnHKStDUsoJnWsKDNGt+CGjLgeIlrMT4HWOpsWRljYfSr4pqx2DbdZSC8cYO7iD4G43Uodmj7yZfcRac4O1OQpfUPwDP1D9DnolwEoCEumfxLe5SZ57Qs5bCsPcwp6dec5ZysGoHno8Nb2l3bH3XcMF7qh5b2odVahhEEDdNtHlvxHGti4vhhkDEDXGmax9qe2xqyvN5iG80ElcCOGx4ewi76hROfD/BhzNkHjp+SYBudTssI8U3WZMv5mamg4Xvds5bwHS5LZm16xliy105fMl2rjcZtlmOe1uo5Fvn02N1Eflt2o236ePHHJUSCjijOzp+l8pL5sxQ8PtDCH33R4gRySmP2+Sz86Jur0UDkAH7uBC2ydMy3g8gFLRqv1BIZ3hhpB8iqZ+GsYrwylGhs3/d8isTP45WNLfJxabHXJAv+dmABSTe0xMj2yUrRXVm5ZScaElnT94+lyUaSCkcup140A6dBqbcnatzmx+KNrVavaSyvLCxEPwbzgmM9F5kY2xoK8FJVtr5dondK9A1YH2uXTKTfUdQveX54fE9BjusK+s/xbQWd44aATyRldhbkpIIc315QoG8Qxy+JnnLFmQgPk1TjFOdsbm3B7tJl22h541YsQ0UuWrCKFqVuToRufNLcGZRoeWUXfkLu8RJx209OnDi11eq0jTXHbvcO4p+r9iI6hi3yvWP7LdsykPHHi11Mt2UfpblgCXvXdhZRSkDyej94dERbvPboMNMN2hrVDZDZSuAJLoybixg9wkODXgcvkuyTDjoGsu/90w3PXUlodiIiGq8jDX2nT8+9ZoiPGemPRuaKjuZ4Gd1XQmfzhBsCdJad6/mdhK7w6W7qb5VLdgLib6fXTlwc4rqwu7aZRnCSx57smW7oXGcbJ+f6S7quafXnMwouWoK+YjcSB+xaQzlwrIybs/t0E/MzffLRALivPIidNc0g7C8PnE6t4bX7S9JWu9dfEWvlC/kUlzbRbJ5lt0PTcKkCz7BoxBG1OwBlgm9bvYbdX15QTvvKb4pfbC/3l6yNXgcZtL6StdFd79NTz12pWf0ZsYi1Vvv2umH2WWSiQUBfrQWadkEd9Riu9QPjZq+/sJh+iza9fNay6Pq3lWZjo7idFqaVMJN/HXM53sKGm2aj5Xs9tI0q9Nf30GCuYO+mddnyu5+hQhMCHjp9Fa2XDTZiasWxV6PX7yACu4HuezTCkzVjEtnHi/PyaiTfnpGfnCPiyZx4xrgYoqdZ4ioSVwti0O8p2JMp2JMp2JM67EmNJQY9idJkMptW7kqzOwXIO2uxfGdG7pycMtYC/Nqiwm1jF7tyirqxFjnZUdAQed5FLXGRf0nNYHhVM2B5lSnvYPmmjHyxWZ2J0s1s4w1wuKagrdinY3HUG9xMHrBd8Lq4V3wmFuEBIB62po9IS+pJuIT6w120mIrsscHd7S15nWuvkspKHTWMjm0l9SEkGzcQtTkjwP+Q+ASrDmq88O6RDUTSW4xUcffNXgvPyEgivdPuoY0LaMAQSVC3PjpF+bREUrWIa5Q+KMKnRk891zor+tm1kUkJnTPSZ3stPCV5ipygzQskDZJOT2Op5y4TZ2h3S9f3ord5XbIT6PToCScu8cdFLTB5tZtYjDOjZ/IRLSBp0em1iXsnwNZlU/wQv98iwSFTDWS8bYdnRjLfCladYGlj+ui0Op4TuSFbv/B2i55jjVIytOmth/ePNTzXQiWHZDnJajzGQqUl9E0nDM5IhCiCJMmaTrtNghpFB6c7iZLV60Y/nBXiYQNZ1m69hjYIOPXatoIQ/UsKEB4qVEnjubQ3EpHANj1/1fQtPDpo1mukopJ9WrGKLHYgqbPYRgNM57qo5KDqFL9qkEiiPURxk7XuNpZ8D31TGwP16cjTROwQBPl2K0mpcR5AtiobDa/nhiPl4A4GshOSo5cPQ4pM5oNOUXk+ACmi8Aa9btdDn4G8Lhi9PIZORwdTMwLJWR5eNxgtgVCc9pcwZCftRAll2UHDd/An1TJ0sdds2j6uC7xI2CuoArj2Ki+ABELpC5Y+Uo6xidrF1INJTgUySanD/YDI8KL6cbQfmJztifAj/eFOy8UrUMro5NDQpu2D6RfBafXjpULH7qQgz7t0HgkseRSFplDBKhExKEuj36BJiDByngQIUNsbs8WMbM9FEbLRhsZ45moQgKJtoTiHID2ZD2t4XccOdgB6FIuDGblvt9Bd3+toohb9sv1YNcrg8NxxJD7KYEDxeOXqvJx06fOTiTFTNjeeATdl/C4aElzpgNwcZFM4Mw+w9XSh3pfF0O5UqqkZAbS5hgZAWjlkN4yguG4vqlAcx4riook/UA4hA7+fiUWHtmAFEI18IzDMRVBsRrkEaQCOcBl0SAltgMa5NNVY8EHKx/OYYLq1vYTYyQKmigqOp5HK5TiKpQ3PbTqt/0p9A4oDbfDRpgkvCIGyBbVkE+VY3I4dLkfTVuxQP3DUhl3CZzkZAGAloaQth+GWxJ6GgZpRaDkHmAweVbCzJNOsAqYjwnB1h1pVGmr14RFlhYHaFBvh0nSPCHa0sR3P9reIyVrHCZxtOVCgsl6RPh1lzXcBat9ueL61HdC49uoOQIzeszcjt+wV2hJtLyjxMGJ/QZykptchdq8b+lmfG9HABh2B5NvNQVCZTgcMgHo3InKd+FhtLqKJxbjzcrjAoFtBGstg7Z/sB6ZrQaXEQVqKJ0pIqiwXUyU7ONsK6YcKQjL6Qmf94iwe4QHkq2AxPWmEDOOKWZIZ5Q0X1aS1baA5WbSSZB9ING0TnZ0SbCtoUdiLASt2HvaXMsXBIUXRXYdiJPI9h2JiFroFR3kIPk4myvoAF6uDPJyqjFyOKqbFch9xUCUtFmNWgS9WZ4ospkqhGhRzCTL5TI9SY14s77kxVDH96Zh3w1zVLxaxYtvADBb6rEOCVSzdtOnPviSu+UGmMz8IMql12MnSZ/uhJNPQGV3xJa043qgOkMML+6UH+CCZC8TfW8jE6YZYMIt+LCw0uj00pEO2Lrn6jkz0HVtEV67heXPS4SPUubfYXW7Gru8XzxgX36oXJ+5u7Yunb+WLp/+nL566lS+e+p+/+Nbl8dT/NI9rtzLGtf9pjKu3MsbV/2GM0eerW/Pi1N2tfDH6DHZrXpy6y724CnhAN5zYgC8sUH0AKKiAC2o6g4REa9lhY9XaTFpvy0F3CfnLaPfDZnRcE1qCiTuU6PNw5cyW3emYU/HpQ1tai24zPtswWsN6FpY1lkxk980gIM9mIz60aWszSL+yocViqAU6LT4f0enY26Pfvu2jxcLxYXFbIzE5pCvSjkVC6rhG3w66nhvYhmc6KFHNcFPT820THdYWLyzeGEvQngnUs9CandBodofxerx44arb65D+GRLGS/Wi0xDJgVRks4bjurZ/uu2iBaJG6HWN5V12Z7XV7d3FvcLzly/wXPt8D23oCm1rR6y53I2OzLMuROs1B2L53dDZmkF4Ae6wJM4Gc+rbkcPOEn3ywkvN7u3QICnR7Ik1xMdLvCBMdDuJ7nzcAb47XkZwqWf12vZmWkEcZkTn4RK1x+6QrpntkuM5cV+n7XScMNhb0JHlIvjmvjOtdfd8VKRWTpxwHOf05NE54ZxGPZyV/nZOOCdOnJl5pt05J7ZRD45zAv/PcTblpc5ZOUn22TmxM/uc/m8rpNgCCDcXZZsKogLUdDbmRbmgOWdmn7PqZk7dPCPzSKdVhmw2Mx41m80M26QfLCpz6IxqUr+tU9PfpyQ/kzeecDI5jh7PSh9wDpyReT6dfjqNejg1/Z28J3WLstahnyifHLqYOc6JLelvhxQbKnEj2absM52IRJJ14NDvdnLOcTQzejpszomN1AN+2ZkZQcapc2Ir/RQX/LNywt3Z5wQ84Tg7WaqcH9lQUBXOSV9MFeJYmhc4m3KCZi6wzokc4ZxwBnMSnMLNpHo1N+f1zomxgqjgzGnuhaD4f3s4yt1s3S6maidLsx1W7ADFTWcbKAc9aTpbIfEWQFhMQyAATfy/ItrM5xnwjiYkaxbDzXhHs7kxL8oVnHzha+Yj0IzzPluUc49Wpn5lX9LMPzebZ2af6Zqc8biZLejNnNNms3lG5pH2KPNgUbaLjjKJ3Rm0gHbXbFKvtxzamjqU1abFTco35Pvp9BP90EydNxNjnHrUdE6cnv6m/Gk6TuqU8tKiY2ZFdfY0SnBq+jvlotwdxGdN09d1xKduh75tdoLtWE8WAZATqdHRw8G2vBjN/BVhcvzxKNmgAfcUo80bp9OMbrUbbS+wt1p2stoJ37aO1u0VhVVjcktOiIb8BdmMcXFBNg3KqoCs6N8U4HYKcDsFuK0DbmugrOi2WkwEvEQxL8TrE7flhGRxIiR129sLUrwsMe8vXpNYZPGCxKLHaDVi0YeaAQirsLCzx7LxcAFfL4APZiZDM7wuLd3fhPYSHT9Q2WzZDQefSoIXDi/b6yhUkci3yYdC9PnlrETaXUflfUvyjBcM4LN3U9myvWL4tmvZfvoCdE5HF513sBkfVd4i/ffQRgvyhhpk5JCcqByd/4u32qIR0X4mEKA5R89YXA/t4AiT6vreYmCQT6LxGLPFpYH7alK+5FNqFtwSLUIOfceOT6zdGcuSrzIBuR1lKFIEXbvRa5uhs2Kniz5Rwh8AgGDdDZcMH12vEB/jfhDA8Hwn2eq3sIAv0RivVMq45IKy4+UkTz/KUpOZY7x6dQeDuRSSd7ptw0VDfyPAh5pEO8PLsPHKof58s3yzGd6pP7Yf6g58qONZpTHoeNZ4ZbwPf3DYL+gDLH9fH54s2i3HBQtSDiTXalxcQppdY7k8KRA1Xpnoyy+cHHfsC+0D6u+dOFFKilqEkmQp8Ta6u6ifqMSXH5VD/b0TR+UIC8UhSmpCdNUEM+JZml/Hsmw/VJ+vxRFixp2wnbCbxIZZCSiUX+UosBTp5238KkeBpGydzSdts9W269TJdyWZHfNR8vTpe3yJb1kuxXw/hSNm+6H6fC1O2sP9sSR1S1LLwvuR+q0aMd1P7GO2H6rP1+LY1xks2yqPV0YATbq6Ap+Cey2AQEvWUqy8n0WBOwHvcXSGAYXp2waZxHcW2/YuiMB5OxZr0JUZaE4efZTtRgt14+slQCh3p1YwSEPRV5PoizDq2/H06JtHsBvQRxc5HM6pomtATPRpmdxk5UYXcaGTDMvh/TnEt83Ac1HEFnsWHo56fsPekqPwzS45WcsO8xj69J+XoRHB1rys7bl2LtLERKGjD8MgzjPf7nghKWu4r+qGexNNHOzoYw7eU8FSJjmxr0BEscbr64qeR1rklKlEaVN8c6RMDCPTOUqzOZayEa4lJgBSMuODU5ipxTcqDTC1uKjEo5Xo01ljyW4sdz2HnOWDDrutsQGWZj/HCVruYVjoEsg4YF27Fd3+RuZS8EFqZqvCUUcf16LCMsoj8TZsr4QhV9YN8xh8v8wgh0B32vGi5NvNAxw1/mhIzkvkvQV9MB3h6Mn+FG7ampa1t6CmbnuZ4CiPHRuPTwhB9nofBz10Ti2vpa4EY2iupBRo/75bOJgnBfprWSJwT85j07ddMz7bDtZZvU43m5KUEjUPKHA7aDVRkQNuinLk396iGLdGOO3OjZVkHyjdkuEWPk02BjDF1HOc7sxoqHIwkFNkHtOEoQaHyGCQaZzxyk5YnfYyikPYRJJDzG4eMbuX5wVgKXFbZcefZ8FdGV/JZVPkjtwdgAZdCrgTkOO9+5ADNM8GOcCZFxcNPKsT51PoGfgGs7i1JUsoyTzTpkjW9lpGx3TQAfSUBE8Y2v6iFzgh2lbqIbOwhwLiAwLjBcFxjxEbInxqYJLqqSh3PFJ/brIHRN0CN8k5RJW+3KBjlA71RZL1Gv2FJJWgkEz05QafxESjQ8kcKlkjkotfC/dFfSOh8EIOx3OPHx9qeG4QHjg8xEISIDMzXHhEQaocoYNE/cYju1xMxyubMwSO2faMKDlKdE9WjC266xn4JPG5jC62LkNpAkRLZY6hLvJhLCev3wE6HD5nZ0bupLdzFBTJ9UZ5hdvD5xJHx88CLgkwXtleVKCrVffkxdFNx+hCxV15HUp523TzEXKSS0G2ZOV4dnlrRobPcs1niO2bgZ3l7E43XM9Hh3rYzVKMV7Ie4UFXPmQop/MyM0ClLBswHK3LKBFom6NiVm6bIzDuPEET1pRsBy2iJqZja9l04tteN1IS3FOOkfS22Di9UO/E99rx1wdkx3dRKrSzH99ujm+xj4eS+FJTdFApamzji5fjFWujGQh9PIi/aaRr04aLDKlaCXEwTxiNJQ8tZs9xl+c5MEeiM+3KcyQCj2R8jSIWxPsKDNtFF/IY4ZLjLjtuay+Dxqf474aVKBMHYBW5Vbkb7IHVOEMPQTr02compz/GAxR8bmd/LIxZjo8OdSc32CLfhkuxCYhICnRGmoliEB862nRcJ1jaBets25rOaKLrMXuZrywFVWoqAVfLjmuNV4YBIL4jhRBVgPDtJvxa326OV+4DKNB2EtsP1/8rJXUWeAFpHOBQxQ0HFNnYHYnsCEBEzUqM1AHEdnsd+NVIM14ZZLohfo4CenRkh+2H6MV4WHN7PtNzo4pqW/Fl8aDwKN+baOolmiobrxzm43gugBxDGoxXSvw2cDzQoc4dBw1VSvyOcRLuQ33BaHGI15woYZOLte1mWZCjtyNrjmYEynxGFzqQ6hPUStDIFqHLY/Gn/V7bnixxQtamRFdOIAdlKUimRqPkhiwI7mzCJRerxisHARU1CZIk5AEeF81QdsxkLomJkT2XZdQiOoQIMm4J1TSd9jgPIIlJMov7PgIiy1ZOkbyBymoORJMNuGmplLORAeLmBCFxVw0yJvkJJagsYMdwWYh6tlCCR66IHYMMs+muGx6juSA6hjtclRnusA406MllvWiyfryyDyJQYPHs8L1pbdR7KF6+ditbqV1Fz/FUZy4dY028tqdtBtluSQwUZbshzOx2bdfKFEE0PdVxXAdt6e/U0fRKF1cR1J02g7BaYCMAHR+IO7q4W2w6ru2TEfEelouO2R1i6eLXHcwD0UaTPHc4z1HzxSY6XssiHWzclZ5kwt31cAlv4FhHvdd4i0tQCGbqwPY7vTBrLjMAqWvpu+MrydFarfn+XcUbUtA6rGV7vX4LXLZNsjIsm5TpeXNov48bRF3eTsccZXEnV223jo8790dYTAftYg+6pr88xkVabafTsX3mu1AJDELfbBt19ruSUjrEQxrdzvQBFtBud4wTXXRuW8dz1/cxMcdt1ZmBbTc7tehb4gCXmR5kqZedjmMs10v0NWZaLNeMJc93rvNcZiijOBprnTbzNa1uaHhBwPSj5bRMLFyp72cypIa27ZbZWGemB6HGWeq4vKNz8VbqRo35Nsu2u4FtL2OKmTwNb8n27VrHs4GSkLOWWJbxqRO0gCH6UAHJjbxHc4BhOc1muqMPPQVXQcx/pUW583/R5+O51hN7Z+MdXbYfoGso2044yCNMy5pi6rMZkFFdy9T8VxLJzHsffVb87zXs+YKUrIlNy0kRwbv+qQ2WQT5tUQ/AteLjVdzwHjn9fyXomR4DXt9uxJNjewoqdL1vDx2ibO8CdEFvMQj9oodoWY2NuiCQI9I52UFrqJUAuwG5vYYXGmSMTnRXUTSRg2poHHi8vBrvTbzOju+G25rRodGD59pX0sLo2jcwhRfNsL8UjsDY/JlLntHwfDQjni5ipIXx+NAk+zuNtuebeIlzSuc117AUYMhNq8+yEYHbG22na8RrrqPGz1jbg8XkM5aLJ7E81Ibjr2JnUzo0i5qsB2mbK6bRW8IH6KG5VAvvAQ5qDB6NcHx3pW1YS37OySDgBC9wXyJWfQ+gJ+1DuLYb0OE7HOq1fQxVdJEwQ+vUayxVs14bYKhMH3+b4Kmb9dopWI2Cfjfyq2O2bKM3l8vbRFWWtxlwJ+Vls16LFhbiWp9TUHfCDGPdUtyiIierpt/poUJCSgdxTU7GWfd6YW+lnQwxd9E6MzBriWaQ0qwuOUEX3X/mNqjVDql+xVvDXUY/TNT0S4PQ7tapl47SOq/rm66HooT3X6B5+9DvdfZATOR+N61zWm2nm6iGaJXtBrbroTJeBcOF+9VUuMbyujBEX+2XcbQD28BHBIzyIN8Oaq4dcpnoZZM8Bs1nuWSPArpWzUA7HQ5CDlo2/qZk2eQXSumJPriub6NTjPtBo+BWuGht1VyJs7DaB4mmv/DZzDhJ6/24MHuBSY5Ixick84NO3ABlIosC5QkDJrO41KjicpTSdb3Gsh3SWRg2O0bbXEcr98gJEmNcHKgaFGSbro1ucMNHGvAZkglHuAw+aBqPPrteYFdZQYvoKGjDIESnI51YXQebhES3l9aZvrlsU17vyygtq217jXQabITSumjVoo+++dcMKiuGIcTruA5Yt9OxMtmABZYUerScAIdpwFtEh2XY4cp0PPpPRPgWhbP5sENmgHHr61g9s32Iz9tWy07YQS6btc7pcB3MjWiovgJmY8fpeEk6G53ObpYu/8qOl0tbuqXBPQ+w+uGFN1NgUPBUAPW+jLNmh1btz6lMqpgiAamYY2UUqib7OFAuUkgLhg7NIVCh25NTLdemwYyJ+1tgNVnques9k9bS1aTlmy7aaBt0bRv1nCOkWkSm8DcZL4m36a6jrp0ZBHZnsb0+WuoimwitdmcKLklk3mEFjCnR9bhK2CYTpQkmkb1meq49ZUyDLi0vDGgrM5jT1V20PNqE9dHcB+1+CNbXwPoRA1R60CFHq/s9nzY8uzPa1kq7k6i2pao4E1eccGtR2ulsJEJ0EjDpL56eCBpmd0v0EG8wRb1c0t8la8/RL9ttRFok24xlZCF19OtytKKILI+aoNcb+XbLXsstN0IdYzJ1Y5/sOStmGy0IJocUHUu9QeRlt8pX7FO8+Cvr4fm3xsOcH2dk+iRbGqbbQKMvD10/3cDD3s2RzGyQTRvhWnAGnllYdcIl9LgNP4VLNloRg5YWGKEZLO/C0sgtUaJz/xrh2p5IQ1YPZnVbG2Ybry8lM1zk0pG7N2qrUYYFbS8/s5LoyoYpGXAHKU6h2UKXBKbLkDfG3+C6duig6G+Kyp3j4nOKHTc8g0iiD9c7yZNFmmJ0ypW3YqALVnZkFNGRYGawvD2S4003tejGC8/fnBWj/NgWnRAQT9OTEI6hZWEkCXBGo1uWGtH9i4ax6uMR9mgKJctyzzlmGJ3FNAnOSRm63KziU7ZC6sbHwDb9+A34uEI8+YVdu21vqWO6brQM3Vx00FRqtWZMok5caDrtaFWffyxFib8BunoKF3HSRpjtHDKD34AbU2Asu2yvG+j0dN/P3cxZ7dcZvlx2qtfu/0V49efhoelqDb3o7DJnuStH+XwmIu3++Oxa0MOlfBT+eq99fD+G8d2a5FpNcp1rzsu9DAovUz2cVSb1jtwrnIt8v3D7lsDOLYEbfcBQgrLhzIWuYyUwvt11nA+lV72O8EG0YLnkhWSl8n4+FB0qMsqncG5r6AjYmaldi6aDOrEdz64bmbZjwuyFXryXg3xsdYnlQQUwWoFC9HtpFE9j+7axYvqO6YbBHloZWel4NnuM1hU58jTOgkzXbK9fZyfL43ZnlHgJXqyaBt2Roxrw+YeN9CtTdLRhMF6Z6sNVNCmfOrpNH45Mv9XDlzSRFQeJ29lb4jawUX8pRNc4w8nCcIdPn0heWe3PaZC6OMxxEd3+1HMbUeYd4cD0I/m0fuyW0Mn8f7S4+La3xjFu5ly8/6V+K9z3807qFXiWod22o+iOV8B3Jhv00G3xedl4Zb7ETVwB4/W20fpba7wy16fLdIdg0PDQPfRgmeY6xNtq3YY9Xjkbcht9qaIiGEkYSZLw0WIpfNK32Ubdo27XtnCRjM9Zwl0aZJzQcaO2GYS5ns6g2e0a9hra84hHjNdnnscrW8xmiD+m4U8cQa/ZdNa2GtRRlWGbfEjekRHi01+wfC8gxx1idFUg5Mhec8LdgLzhm8GSbWWd2J2uT25D3mUUzs/07SD0fHtHUYOjP0zL/R6qNwY2eE78zoyfZI1mx3Tai97aZsNYNYNOtJQCbz/aFInIACBEd30bRhBajkfO1Do9fkLDyuQBvSbh8HlXOwyjG0U5sNvNpAOpGUanY3Z3GtnoWOjaP+Rl3TAaa2vmorNSRRm/0nGi4RW51wR/ESOdXWPRRgMHKwhvgRtz0UMb6oLweB9ulswA3fnYWXRaPa8XGN3eYttp4IPWa1nnQR+h7N9JEshzyp1wwjiRdY036SYXxJBPi+iKAzQ0OZpFmz3XMlHDYrZB/OwsXhr1Pvkk3gslPCfS4yVOkzjkUgdfRGOjNh8f9o4Hzs1o/DsOo7j+Z8C9GDRabW/RbJPFkIaFNpXMT8/Umcq5yak5pnJ2ZmaGrazVp9nKap3t7czs1DxbOTPNUU7PzLKV9Tl2aGfqtSmOsspWTs/PT3KUVXaApuc4aTs9O1XlKGvsd07NTbJTaGp2jv3O+tzc/B6mcpYdnvrMDMfX6iQ73Wv1KXYpqc5PzuxjKmfmp9iJUJ2ZrnKc1qam2Nqpyck6W1ufm2JnabU+x3ltfXZ6mqOdmZ5na2tz1UmOtl7nvLdWnZ/larlu57hu53ihqs5w3XLjW+WlVa3Ky95adYr73jrX5zo3RjVO0ahVq1yfJ3nxnZzjuZ2c5cVokptWk9y0mpzi1LHaZJ1TUWqTk5xQVefnOaGqzvNKXXWeV+qq87OcUFXnZzk5WJ2f4cS3Oj/N9XmqxtPyylV1nldDq/NVTtmozvPKVXV+kt1+Iy3H6dz8HFfLS4w5nqmrzs3wIjQ3w/V5hhvmaa52ipdUc3VO9a3O1bhhrnF9rnJTY5LrdpLndnaOF6pZnmGozs7w4jvLM8BVbmNVnZ7mlZwpnuGvVrlVYZJXFSbnJjlVcHKGq53mmY3JqSrPba06P8DQHp1HfeiZQaZ63piem57ezdaznc4Z03OTdbbTObZqhq2aYr+wjsLKeSFHVdvFVDEdzbGTbW7amJ+ZnBri6KsztTo7QHPTbLd1VIQnOeHieFtjq6qs7vPRuclpjo6ddOwMnmWHfnaWrZpmJvhs3Zif5pSoWXaazFaZST07ibKpxgnsJDPys8ywzqA+79QkW4/qJKcKzLDr3MycUavPTLGdzrGdzqK3smv6DDtfZjhh5XjICWadHcwaGgyzi/EMR1VlZha7Jk8jy1xlZ9Y0ts3sZJ3GmVllBmp6nln6pufQ0GmS45Zd3qfZxW96xqhXq+wsm55hhwibrRrHLbuGTk+h17LN2vQUim2dnX3TbHM6XcduJ9lu2dV/usb2toq85UWXkzdsy8BOoylUWDjNytS8UatzSszUPA4wOw2nOG5xaaux03CKXcinkO3gNFlTs3gSgF1cp2bZbmew2xlmCzTFMT9T7LI8NY39ZRcqDNR5nnMijAtzlZMR7G7MFGrKOG34VJ392hqqnxyzMFVjv7aKXssJcZXtdBJNp3KKx6QxPTvNifEkMg3sejiFm2GOXUJArc5pazitbR13vdl1so4NPEeP6yzbENTZVa4+h+M1xSza9Vl2ftTZVaY+g/2tsf2dYReROrvA19kFvj6FX8m2HnV2a19nt/Z11NrPc2KKC/wUJzbsUlWvYrec6HIylW3d2ZGpoZIyyzZktXl2F7w2z07aGnPi+2iNPSl+tDY3x9HNcnTsBqqGGgJOZanNot4b572zdY6OXaBrs9Vhtm4SDZg445vaLCdx2R2p2gz60sGuE7UZNDE+x67etRlO7sxwUolj6WqczJlGmcMJzjTb2tQ4eTplTM/XJtnJMIUsbI09hKxNsQdYNXarUUPtJK+81DlR5fQVa5wajNrIWXYbWquir2bs4VmtatTneDWZPRStzuMmjt0t4Ly2OoeCNcf2ew77zXn5HB4Fsc0sz/NZ7JZtZqvsGlZFJXaG3YBWp9FXK06rU2WXumqdbeKqdbaVrnKKVbXOLo5VTnNSZXfpqjVkRGYm2f7WONnCrtBVtjmsTrLNUpUzFVTlTAVVdzA0m5HcNLo9tAzS8cOe2R4hIsdt+DZehZcsOELnQ+Gjq3YQBC3uTZXd0D+LyJPFCFsMoxFdGNBD594Yba9BydreKpFtj2X40uZFdK8uWvI+fMUdLrucXNd7EbkHIisYr1TyxBVOu32B73UvMYPwwjWyNudu6Lw+TJJ7fkvInZdccqlxjzsbF1162SULC9TDmej3xfcksovvKQiiIAmyoAiqoAm6sEE4RThV2CCcJmwQThc2CGcIZwpnCRuFTcIGYbOwRdgqbBO2CzuEncIuYbewR9gr7BMGhA3CoDAkbBCGhRFhVNggjAn7hQPCQWFcqAgTwiHhsHBEOCqcLUwKVaEm1IUpYVqYEWaFOWFeWBBuI2wQjgkbhHOE48IG4bbCucLthPOEDcL5wgXChcIG4fbCHYQNwgbhjsIG4SLhTsLFwiXCpcKdhbsIlwl3Fe4m3F24h3C5cE/hCuFK4Sphg3AvYYNwtXBv4Rphg3CtYGC3G4T7CGb0C/0tCg3BEmyhiZ9a+L9LgiOcEJaFttARNgiu4AldLD8p+EIghMIGoSesCBuEVWFNWBeuEzYI9xXuJ9xfuB5TDxAeiP99kPBg/O9DhIfifx8mPFx4hPBI4VHCo4XHYMljhccJG4THC08Qnig8SXiy8BRhg/BU4WnCBuHpwjOEDcIzhWcJzxaeIzxXeJ7w/Ci8LxBeKLxIeDH+/RLhpfjflwkvF14hvFJ4lfDqiHqN8FrhdcLrhTcINwhvjGRvEt4svEV4q/A24e3CO4R3Cu8S3i28R3gvlRbvE94vfED4oHAjJTsgfEj4sPAR4aPCx4SPR7JPCJ8UPiV8WviM8Fnhc8IG4fPCF4QvCl8Svix8RfiqcJPwNeHrwjeEbwobhG8J3xa+I3xX+J7wfeEHwg8pf8nfj4QfCz8Rfir8TPi58Avhl8KvMtpfC78Rfiv8Tvi98Afhj8KfhD8LfxH+Kvwt58ffhX8I/xT+Jfxb+I9wsyCIonjzzalWEmVREVVRE3Vxg3iKeKp4mni6eIZItGeKZ4kbxU3iZnFA2CJuFbeK28TY5XZxh7hT3CXuFpeEPeJeMR929LdPHBAHxSFxWBwRR7H7JWFM3C8eEA+K4wUXFXFCPCQeFo+IR8WzxUmxKtbEujglTosz4qw4J86LC+JtxGPiOeKNwnHxtuK54Ds3CLcTzxPPFy8QLxSvF24v3kG8o3iReCfxYvES8VLxzuJdxGPiZeJdGW7vJt5dvId4uXhP8QrxSvEq8V7i1eK9xWvEa0VDvI9oiotiQ7REW2yKLXFJdMQT4jHxRmFZbDP8Q38d0RU9sSt2xZOiLwZiKPbEFXFVXBPXxevE+4r3E+8vXi8+QHyg+CDxweJDxIeKc+LDxIeLjxCPiY9kxpL8PUp8tPgY8bHi48THi08Qnyg+SXyy+BTxqeLTxKeLzxCfKT5LfLb4HPG54vPE54svEF8ovkh8sfgS8aXiy8SXi68QXym+Sny1+BrxNeJrxdeJrxffIN4gvlF8E/eNrL83i28R3yq+TXy7+A7xneK7xHeL7xHfK75PfL/4AfGD4o3ih8QPix8RPyp+TPy4+Anxk+KnxE+LnxE/K35O/DwuGV8Qvyh+Sfyy+BXxq+JN4tfEr3NC8Q3xm+K3xG+L3xG/K35P/L74A/GH4o/EH4s/EX8q/kz8ufgL8Zfir8Rfi78Rfyv+Tvy9+Afxj+KfxD+LfxH/Kv5N/Lv4D/Gf4r/Ef4v/EW8WBUmUJEmWFEmVNEmXNkjHxOPiKdKp0o3CadLp0q1Ji/7+zpDOlM6SNkqbpM3SFmmrJAsVcZu0Tdou7ZB2Sruk3dIeaa+0TxqQBqUhaVgakUalMWm/dEA6KI1LFWlCOiQdlo5IR6WzpUmpKtWkujQlTUsz0qw0J81LC9JtpJuEY9I50nHpttK50u2k86TzpQukC6XbS3eQ7ihdJN1Juli6RLpUurN0F+ky6a7S3aS7S/eQLpfuKV0hHROPiVdKV0n3km4Qr5buLV3zfzAlyN+1kiHdRzKlRWlRakiWZEtNqSUtSY50QlqW2lJHciVP6koT4knJlwIplHrSirQqrUnr0nXSfaX7SfeXrpceID1QepD0YOkh0kOlh0kPlx4hPVJ6lPRo6THSY6XHSY+XniA9UXqS9GTpKdJTpadJT5eeIT1Tepb0bOk50nOl50nPl14gvVB6kfRi6SXSS6WXSS+XXiG9UnqV9GrpNdJrpddJr5feIN0gvVF6k/Rm6c3SW6S3Sm+T3i69Q3qn9C7p3dJ7pPdK75PeL31A+qB0o/Qh6cPSR6Sbb/6odJXwFfFj0selT0iflD4lfVr6jPRZ6Ubhc9LnpS9IX5S+JH1Z+or01f/j6fu//3eT9DXp69I3pG9K35K+LX1H+q70Pen70g+kH0o/kn4s/UT6qfQz6efSL6RfSr+Sfi39Rvqt9Dvp99IfpD9Kf5L+LP1F+qv0N+nv0oT4D+mf0r+kf0v/kW6WBFmUJVmWFVmVNVmXN8inyKfKp8mny2fIZ8pnyRvlTfJmeYu8Vd4mb5d3yDvlXfJueY+8V94nD8iD8pA8LI/Io/KYvF8+IB+Ux+WKPCEfkg/LR+Sj8tnypFyVa3JdnpKn5Rl5Vp6T5+UF+TbyMfkc+bh8W/lc+XbyefJbpfPlAeEC+UL59vId5DvKF8l3ki+WL5EfJFwq31m+i3yZfFf5bvLd5XvIl8v3lK+Qr5R3CVfJ95Kvlq+W7y2fKl0jXysb8n1kU16UG7Il23JTbslLsiOfkJflttyRXdmTuzKqlejvGvmkfFJ+pejLyIIHciBfI4cy+ve4eKV0Cq631wg9+bPSinyjsCqvyevydfJ95fvJ95evlx8gP1D+3y4L///8PUh+sPwQ+aHyw+SHy4+QHyk/Sn60/Bj5sfLj5MfLT5CfKD9JfrL8FPmp8tPkp8vPkJ8pP0t+tvwc+bny8+Tnyy+QXyi/SH6x/BL5pfLL5JfLr5BfKb9KfrX8Gvm18uvk18tvkG+Q3yi/UX6T/Cb5zfIB4S3yW+W3yW+X3yG/U36X/G75PfJ75ffJ75c/IH9QvlH+kPxh+SPyR+WPyR+XPyF/Uv6U/Gn5M/Jn5c/Jn5e/IH9RnhC/JH9Z/or8L+mr8k3y1+Svy9+Qvyl/S/62/B35u/L35O/LP5B/KP9I/rH8E/mn8s/kn8u/kH8p/0r+tfwb+bfy7+Tfy3+Q/yj/Sf6z/Bf5r/Lf5L/L/5D/If9T/pf8b/k/8s2yoIiKpMiKoqiKpujKBuUU5VTlNOV05QzlTOUsZaOySdmsbFG2KtuU7coOZaeyS9mt7FH2KvuUAWVQGVKGlRFlVBlT9isHlIPKuFJRJpRDymHliHJUOVuZVKpKTakrU8q0MqPMKnPKvLKg3EY5ppyjHFduq5yr3E45TzlfuUC5ULm9cgflJuGOykXKnZSLlUuUS5U7K3dRLlPuqtxNubtyD+Vy5Z7KFcqVylXKvZSrla5wb+Ua5VrFUO6jmMqi0lAsxVaaSktZUhzlhLKstJWO4iqe0lVOKr4SKKHSU1aUVWVNWVeuU+6r3E95j3R/5XrlAcoDlQcpD1YeojxUeZjycOURyiOVRymPVh6jPFZ5nPJ45QnKE5UnKU9WdglPUZ6qPE15uvIM5ZnKM5U/Cc9Snq08R3mu8jzl+coLlBcqL1JerLxEeanyMuXlyiuUVyqvUl6tvEZ5rfI65fXKG5QblGV5WX6j8ibl5ptvvvnNyluUtypvU96uvEN5p/Iu5d3Ke5T3Ku9T3q+8X/mA8kHlRuVDyoeVjygfVT6mfFz5hPJJ5VPKp5XPKJ9VPqd8Xvm88gXli8qXlC8rX1G+qtykfE35uvJ15RvKN5VvKcfEbyvfVlBv5SrpO8p3qb9Xit9TfPn7yg+UV4m2eFwM5VeL31WQzfih8iPlR8qPlZ8ogUz+Xil+SjpV+qnyPeXTEnHzGYm4erX4M+Xnyi+UXypXSb9SfqX8Wvm18hvlt8rvlN8rf1D+qPxJ+bPyF+Wvyt+Uvyv/UP6p/Ev5t/If5WZFUEVVUmVVUVVVU3V1g3qKeqp6mvq/bQ3+39//nX+nq2eoZ6pnqRvVTepmdYu6Vd2mbld3qDvVXepudY/6dnGvuk8dUAfVIXVYfaYwoo6qY+p+9YB6UB1Xx9WKOqEeUg+rR9Sj6tnqpFpVa2pdnVKn1Rl1Vp1T59UF9TbqMfUc9bh6W/Vc9Xbqeer56gXqhert1Tuod1QvUu+kXqxeol6q3lm9i3qZelf1burd1Xuol6v3VK9Qr1SvUu+lXq3eW71GvVq9VjXU+6imuqg2pYZqqbbaVFvqkuqoJ9Rlta12VFf11K56UvXVQA3Vnrqirqpr6rp6nXpf9X7q/dXr1QeoD1QfpD5YfYj6UPVh6sPVR6iPVB+lPlp9jPpY9XHq49UnqE9Un6Q+WX2K+lT1aerT1Weoz1SfpT5bfY76XPV56vPVF6gvVF+kvlj9m/QS9aXqy9SXq69Q3ye+Un2V+mr1Nepr1depr1ffoE6IE+IN6knpjeqb1Derb1Hfqr5Nfbv6feUd6jvVd6nvVt+jvld9n/p+9QPqB9Ub1Q+pH1Y/on5U/Zj6cfUTqih/Uv2k+in10+pn1M+qn1M/r35B/aL6JfXL6lfUr6o3qTepX1O/rn5D/ab6LfXb6nfU76rfU7+v/kD9ofoj9cfqj9WfqLL8U/Vn6s/VX6i/VH+l/lr9jfpb9Xfq79U/qH9U/6T+Sf2z+hf1r+rf1L+r/1D/qf5L/bf6H/VmVdBETdJkTdFUTdN0bYN2inaqdpp2unaGdqZ2lrZR26Rt1rZoW7Vt2nZth7ZT26Xt1vZoe7V92oA2qA1pw9qINqqNafu1A9pBbVyraBPaIe2wdkQ7qp2tTWpVrabVtSltWpvRZrU5bV5b0G6jHdPO0Y5rt9XO1W6nnaedr12gXajdXruDdkftIu1O2sXaJdql2p21u2iXaXfV7qbdXbuHdrl2T+0K7UrtKu1e2tXavbVrtGs1Q7uPZmqLWkOzNFtrai1tSXO0E9qydo3c1jqaq3laVzup+VqghVpPW9FWtFVtTVvXrtPuq91Pu792vfYA7YHag7QHaw/RHqo9THu49gjtkdqjtEdrj9Eeqz1Oe7z2BO2J2pO0J2tP0Z6qPU17uvYM7Znas7Rna8/Rnqs9T3u+9gLthdqLtBdrL9Feqr1Me7n2Cu2V2qu0V2uv0V6rvU57vfYG7QbtjdqbtDdrb9Heqr1Ne7v2Du2d2ru0d2vv0d6rvU97v/YB7YPajdqHtA9rH9E+qn1M+7j2Ce2T2qe0T2uf0T6rfU77vPYF7Yval7Qva1/RvqrdpH1N+7r2De2b2re0b2vf0b6rfU/7vvYD7Yfaj7Qfaz/Rfqr9TPu59gvtuPhL7Vfar7XfaL/VfqfdWzlXOldCswtd4WrlXAn9vyvcIN0gnSv9XvuD9kftT9qftb9of9X+pv1d+4f2T+1f2r+0f2v/0W7WBF3UJV3WFV3VNV3XN+in6Kfqp+mn62foZ+pn6Rv1TfpmfYu+Vd+mv026QN6u79B36rv03foe/fbSXr2r7NMH9EF9SB/WR/RVZVQf0/frB/SDuimM6xV9Qj+k31c5rB/Rj+pn65N6Va/pdX1Kn9Zn9Fn9gcqcPq8v6LfRj+nn6Mf12+q31c/Vb6efp5+vX6BfqN9ev4N+R/0i/TKZ/N1Jv1i/RL9Uv7N+F/0y/a763fS76/fQL9fvqV+hX6lfpd9Lv1q/t36Nfq1u6PfRTd3UF6k/1PYX233UW1jUG3pDt3RLP0uw9aa+VWjpS7qjn9CX9bbe0V3d07v6Sd3XAz3Ue/qKvqqv6ev6dfp99fvp99ev1x+gP1B/kP5g/SH6Q/WH6Q/XH6E/Un+U/mj9Mfpj9cfpj9efoD9RR/2kJ+lP1lE/6Sn6U/Wn6U/Xn6E/U3+W/mz9Ofpz9efpz9dfoL9Qf5H+Yv0l+kv1l+kv11+hv1J/lf5q/TX6a/XX6a/X36DfoL9Rf5P+Zv0t+lv1t+lv19+hv1N/l/5u/T36e/X36e/XP6B/UL9R/5D+Yf2t2qeVj+gf1T+mf1z/hP5J/VP6p/XP6J/VP6dfIX1e/4L+WeWL+hf1L+lf1r8S/X1Vf714k/41/ev6N/Rv6t/Sv61/R/+ujvpl39O/r/9A/+H/1849BWnSrHkAb739Zj5p1di2e2zbZo9t27Zt27Zt27bZszH9fefE2bO43b3o/MU/I6sqM2+rKqKe8j/wP/Q/+ttj/2P/E/8T/1P/U/+zf/Pc/9z/wv/C/9L/0v/qX7yO9Mb/1v/O/97/wf/B3yb4o//PU94n/yf/9ZD/3l/vgdWCagR99n/xf/V/83/3//B3Cu4U/NP/y3/FF+GP8P+O9K+3wtyBgSgQBaFgFIJ8yIdCkR/50Z+GEUaAABFEEEUUMcSRQBIppJFBBlnkkIeioegoBoqJYqHYKA6Ki+Kh+CgBSogSocQoCUqKkqHkKAVKiVKh1CgNSovSofQoA8qIMqHMKAsKQ1lRNpQd5UA5US6UG+VBeVE+lB8VQAVRqK8QKoyKRCqKiiHsK45KoJKoFCqNyqCyqBwqjyqgiuj/+lkiSpQoUaJEiRIlSpQoUaJEiRIlSpQoUaL8f1EJVUZVUFVUDVVHNVBN5PlqodqoDqqL6qFwVB81QA1RI9QYbQlsgpqiZqg5ao5aoJaoFWqFWqM2qA1qi9qh9qgD6og6ok6oM+qMuqCuqCvqhrqj7qgH6ol6oV6oN+qD+qC+qB/qh/qjAWgAGoji+gahwWgIiu8bioah4WgEGolGodFoDBqLxqHxaAKaiCahyWgKmoKmomloOpqBZqJZaDaag+aiuWgemo8WoIVoEVqMlqClaBlajlaglWgVWo3WoLVoHVqPNqBUvo1oE9qMtqCtaBvajnagnWgX2o32oL1oH9qPDqAD6CA6hA6jI+goOoaOoxPoJDqJTqHT6Aw6ix4EnUPn0QV0EV1Cl9EVdBVdQ9fRDXQT3US30G10B91F99B9dB89QA/RI/QYPUFP0TP0HL1AL9Er9Bq9QW/RO/QefUAf0Sf0GX1BX9E39B1NDv6BfqJfKAL9RgE4EAfhYByCfTgU+zHC+XwYAyaYYIoZ5lhgiRXW2GCLHfZwNBwdx8AxcSwcG8fBcXE8HB8nwAlxIpwYJ8FJcTKcHKfAJXwpcSqcGqfBaXE6nB5nwBlxJpwZZ8FhOCvOhrPjHDgnLuvLhXPjPDgvzofz4wK4IC6EC+MiuCguhovjErgkLolL4VK4NC6Dy+JyuBwujyvgin/3lXBlXAVXxdVwNVwd18A1cS1cG9f5T+O6eH1wXV89HI7r4waRGuJGuDFugpviZrg5boFb4la4NW6D2+J2kdrjDrgj7oQ74y64K+72t+64B+6Je+HeuA/ui/vh/ngAHogH4cF4CB6Kh+HheAQeiUfh0XgMHovH4fE4PGACnogn4cl4S/AUPBVPw9PxDDwTz8Kz8Rw8F8/D8/ECvBAvwovxErwUL8PL8Qq8Eq/Cq/EavBavw+vxBrwRb8Kb8Ra8FW/D2/EOvBPvwrvxHrwX78P78QF8EB/Ch/ERfBQfw8fxCXwSn8Kn8Rl8Fp/D5/EFfBFfwpfxFXwVX8PX8Q18A9/Et/BtfAffxffwffwAP8SP8GP8BD/Fz/Bz/AK/xK/wa/wav8FvcVffO/wef8Af8SfczfcZd/d9wV/xN/wdf8c/8E/8E//CETgC/8YBEAiBEATBEAI+CIVQ8AMCDBgACBCgwICDAAESFGjQYMBCL58DD6JBdIgBMSEWxIY4EBfiQXxIAAkhESSGJJAUkkFySAEpIRWkhjSQFtJBesgAGSETZIYsEAZZIRtkhxyQE3JBbsgDeSEf5IcCUBAKQWEoAkWhGBSHElASSkFpKANloRyUhwpQESpBZagCVaEaVIcaUBNqQW2oA3WhHoRDfWgADaERNIYm0BSaQXNoAS2hFbSG1tAG2kI7aA/toQPM93WETtApoDN0hi7QFbpBd+gBPaEn9ILesMDXB/pCP+gPA2AgDIJBMBiGwFAYCsNgOIyAkTAKRsFoGANjYRyMhwkwESbBZJgMU2AqTIPpMANmwEyYBbNhNsyBuTAP5sF8WAALYSEsgsWwBJbCMlgOK2AlrILVsAbWwjpYB+thA2yEjbAJNsMW2ArbYBtshx2wE84H74LdsAf2wj7YB/vhACzyHYRDcAgOwxE4CkfhGByHE3ACTsIpOA2n4QychXNwHlIEXICLcAkuwxKfL/AKXIVrcB1uwE24BbfhDqz03YV7cB8ewEN4BI/hCTyFZ/AcXkDKwJfwCl7DG3gL7+A9fICP8BE+wWf4Al/hG3yHH/ATfkEE/IYAEkiCSDAJIT4SSvwEEUziBQEhhBJGOBFEEkU0McQSRzwSjUQnMUhMEovEJnFIXBKPxCcJSEKSiCQmSUhSkowkJylISpKKpCZpSFqSjqQnGUhGkolkJllIGMlKspHsJAfJSXKR3CQPyUvykfykAClICpHCpAgpSoqR4qQEKUlKkdKkDClLypHypAKpSCqRyqQKqUqqkeqkBqlJapHapA6pS+qRcFKfNCANSSPSmDQhTUkz0py0IC1JK9KatCFtSTvSnnQgHUkn0pl0IV1JN9Kd9CA9SS/Sm/QhfUk/0p8MIAPJIDKYDCFDyTAynIwgI8koMpqMIWPJODKeTCATySQymUwhU8k0Mp3MIDPJLDKbzCFzyTwynywgC8kispgsIUvJMrKcrCArySqymqwha8k6sp5sIBvJJrKZbCFbyTaynewgO8kuspvsIXvJPrKfHCAHySFymBwhR8kxcpycICfJKXKanCFnyTlynlwgF8klcplcIVfJNXKd3CA3yS1ym9whd8k9cp88IA/JI/KYPCFPyTPynLwgL8kr8pq8IW/JO/KefCAfySfymXwhX8lX8o18Jz/IT/KLRJDfJIAG0iAaTEOoj4ZSP0UUU6CEUsoop4JKqqimhlrqqEej0eg0MDgGjUlj0dg0Do1L49H4NAFNSBPRxDQJTUqT0eQ0BU1JU9HUNA1NS9PR9DQDPenLSDPRzDQLDaNZaTaaneagOWkumpvmoXlpPpqfFqAFaSFamBahRWkxWpyWoCVpKVqalqFlaTlanlagFWklWplWoVVpNVqd1qA1aS1am9ahdWk9Gk7r0wa0IW1EG9MmtCltRpvTFrQlbUVb0za0LW1H29MOtCPtRDvTLrQr7Ua70x60J+1Fe9M+tC/tR/vTAXQgHUQH0yF0KB1Gh9MRdCQdRUfTMXQsHUfH0wl0Ip1EJ9MpdCqdRqfTGXQmnUVn0zl0Lp1H59MFdCFdRBfTJXQpXUaX0xV0JV1FV9M1dC1dR9fTDXQj3UQ30y10K91Gt9MddCfdRXfTPXQv3Uf30wP0ID1ED9Mj9Cg9Ro/TE/QkPUVP0zP0LD1Hz9ML9CK9RC/TK/QqvUav0xv0Jr1Fb9M79C69R+/TB/QhfUQf0yf0KX1Gn9MX9CV9RV/TN/QtfUff0/f0A/1IP9HP9Av9Sr/R7/QH/Ul/0Qj6mwawQBbEglkI87FQ5meIYQaMMMoY40wwyRTTzDDLHPNYNHbBF53FYDFZLBabxWFxWTwWnyVgCVkilpglYUlZMpacpWApWSqWmqVhaVk6lp5lYBlZJpaZZWFhLCvLxrKzHCwny8VyszwsL8vH8rMCrCArxAqzIqwoK8aKsxKsJCvFSrMyrCwrx8qzCqwiq8QqsyqsKqvGqrMarCarxWqzOqwuq8vqsXBWnzVgDVkj1pg1YU1ZM9actWAtWSvWmrVhbVk71p51YB1ZJ9aZdWFdWTfWnfVgPVkv1pv1YX1ZP9afDWAD2SA2mA1hQ9kwNpyNYCPZKDaajWFj2Tg2nk1gE9kkNplNYVPZNDadzWAz2Sw2m81hc9k8Np8tYAvZIraYLWFL2TK2nK1gK9kqtpqtYWvZOraebWAb2SZ21beZbWFb2Ta2ne1gO9kutpvtYXvZPrafHWAH2SF2mB1hR9kxdpydYCfZKXaanWFn2Tl2nl1gF9kldpldYVfZVXaNXWc32E12i91md9hddo/dZw/YQ/aIPWZP2FP2jD1nL9hL9oq9Zm/YW/aOvWcf2Ef2iX1mX9hX9o19Zz/YT/aLRbDf7ElwAA/kQTyYh3AfD+V+jjjmwAmnnHHOBZdccc0Nt9xxj0fj0XkMHpPH4rF5HB6Xx+PxeQKekCfiiXkSnpQn48l5Cp6Sp+KpeRqelqfj6XkGnpFn4pl5Fh7Gs/JsPDvPwXPyXDw3z8Pz8nw8Py/AC/JCvDAvwovyYrw4L8FL8lK8NC/Dy/JyvDyvwCvySrwyr8Kr8mq8Oq/Ba/JavDavw+vyejyc1+cNeEPeiDfmTXhT3ow35y14S96Kt+ZteFvejrfnHXhH3ol35l14V96Nd+c9eE/ei/fmfXhf3o/35wP4QD6ID+ZD+FA+jA/nI/hIPoqP4qP5GD6Wj+Pj+QQ+kU/ik/kUPpVP49P5DD6Tz+Kz+Rw+l8/j8/kCvpAv4ov5Er6EL+XL+HK+gq/kq/hqvoav5ev4er6Bb+Sb+Ga+hW/l2/h2voPv5Lv4br6H7+X7+H5+gB/kh/hhfoQf5cf4cX6Cn+Sn+Gl+hp/l5/h5foFf5Jf4ZX6FX+XX+HV+g9/kt/htfoff5ff4ff6AP+DBwQ/5I/6YP+FP+TP+nL/gL/kr/pq/4W/5O/6ef+Af+Sf+mX/hX/k3/p3/4D/5Lx7Bf/MAESiCRLAIET4RKvwCCSxAEEEFE1xwIYQUSmhhhBVOeCKaiC5iiJgilogt4oi4Ip6ILxKIhCKRSCySiKQimUguUoiUIpVILdKItCKdSC8yiIwik8gssogwkVVkE9lFDpFT5BK5RR6RV+QT+UR+UUAUFIVEYVFEFBXFRHFRQpQUpURp4Qv+o4woK8qJ8qK8qCAqikqisqgiqopqorqoIZ77aopaoraoI+qKeiJc1BcNREPRSDQWTURT0Uw0Fy1ES9FKtBatRZtIbUVb0U60Fx1ER/HS10l0jtRFdBXvfd1Ed9FD9BS9RG/RR/QV/UR/MUAMFIPEYIGD//rmdogYKoaJ4WKEGClGidFijBgrxonxYoKYKCaKSWJypCliqpgmponpYoaYKWaJ2WKOmCvmifligVgoFonFYolYKpaJ5WK5WBFppVgpgkJXidVijVgrgkPXifWRNoiNYpPYLLaIrSIkdJvYHmmH2Cl2id1ij9gr9on9whd6QBwUh8RhcUQcFcfEcXFCnBSnxGlxRpwVZ8W5SOfFeXFBXBSXxGVxRVwVV8W1SNfFdXFD3BS3xG1xR9wV98R98UA8FI/EY/FEPBXPxHPxQrwU/tBXomrga/FGvBXvxHvxQXwUKPST+Cy+iK/im/gufgb/ED/FL/FLRIjfIkAGyiAZLEOkCvbJUOmXSCKJJUgiqWSSSyGlVFJJLbU00konPSlDo8loMrqMIWPImDKWNKGxZRwZV8aT8WUCmVAmkollEplUJpPJZQqZUqaSqWUamVamk+llBplRZpKZZRYZJsNkVplNZpc5ZE6ZS/pCcss8Mq/MJ/PLArKgLCQLyyLyQGBRWUwWlyVkSVlKlpZlZFlZTpaXFWRFWUlWllVkVVlNVpc1ZE1ZS9aWdWQdWVfWk+GyvswV2kA2lI1kY9lENpXNZHPZQraUrWRr2Ua2kW1lO9lOtpcdZEfZSXaWXWRX2U12lz1kT9lL9pZ9ZP7Q/KF9ZT/ZXw6QA+UgOVgOkUPlMDlcDpcj5Eg5So6WY+QYOVaOk+PlBGlCJspJcrKc8repcpqcLmfImXKWnC3nyDlyrpwn58sFcqFcJBfLJXKJXCqXymVyuVwhV8pVcrVcLdfItXKdXC83yI1yk9wsN8stcqvcJrfLHXKn3CV3y60Be+TeSPvkfnlAHpQH5SF5WB6RR+VReUwelyfkSXlKnpKn5Rl5Vp6T5+R5eUFelJfkZXlZXpFX5TV5Xd6QN+UteUvelnfkXXlP3pf35QP5UD6Sj+Rj+UQ+lc/kM/lcvpAv5Sv5Sr6Wb+Rb+U6+lx/kB/lRfpKf5GdZKvSL/Cq/ye/yh/wpf8kI+VsGqEAVpIJViPKpUOVXSGEFiiiiqGKKK6Gk+tO0Msoqp5zyVDQVXcVQ4QHhATFVLBVbxVFxVVwVT8VXCVRClUglVolVEpVUJVPJVXKVQqVUqVRqlUalUWlVOpVeVQ3NoDKqTCpuSGaVRYWpMJVVZVPZVQ6VQ+VUuVRulUflVflUfjUuqIAqoAqqQqqwKqKKqKKqmCqmiqsSqqQqpUqp0qqMKqvKqfKqgqqoKqnKqrKqoqqqapGqq+qqhqqpwgNqqdqqjqqr6qlwVV+1Cf2jgWqoGqnGqolqqpqp5qqFaqlaqdaqjWqr2qn2qoPqqDqHdlKdVGfVWXVRXVRX1VV1U91Ud9Vd9VA9VE/VU/VSvVRv1Vv1UX1UX9VX9fsf9VcD1EA1SA1WQ9RQNUwNVyPUSDVKjVZj1Fg1To1XE9RENUlNVlPUVDVNTVcz1Ew1S81Wc9RcNVfNU/PVArVQLVKL1RK1VC1Ty9RytVytUCvVKrVarVFr1Tq1Xm1QG9UmtVltUVvVNrVd7VA71S61W+1Re9U+tV8dUAfVIXVYHVFH1TF1XJ1QJ9UpdVqdUWfVOXVeXVAX1SV1WV1RV9U11Za1Z9fVDXVT3VK31R11V91T99UD9VA9Uo/VE/VUPVPP1Qv1Ur1Sr9Ub9Va9U+/VB/VRfVKf1Rf1VX1T39UP9VP9UhHqtwrQgX8L0sE6RPt0qPZrpLEGTTTVTB/0cy201EprbbTVTns6mo6uY+iYOpaOrePouDqejq8T6IQ6kU6skwcn0Ul1Mp1cp9ApdSqdWqfRaXU6nV5n0Bl10ZA/MukrAf+bzDqLDtNZdTadXefQOXUunVvn0Xl1Xp1P59cFdEFdSBfWRXRRXUwX1yV0SV1Kl9ZldFldTpfXFXRFXUlX1lV0VV1NV9c1dE1dS9fWdXRdXU+H6/q6gW6oG+nGuoluqpvp5rqFbqFPhbbUrXRr3Ua31e10e91ed9AddSfdWXfRXXU33V330D11L91b99F9dT/dXw/QA/UgPVgP0UP1sMgM1yMiM1KPisxoPSYyY/W4f2a8nqAn6kmRmaynRGaqnhaZ6XqGnqln6Vl6tp6j5+p5er5eoBfqRXqxXqKX6mV6uV6hV+pVerVeo9fqdXq93qA36k16s96it+pterveoXfqXXq33qP36n16vz6gD+pD+rA+oo/qY/q4PqFP6lP6tD6jz+pz+ry+oC/qS/qyvqKv6mv6ur6hb+pb+ra+o+/qe/q+fqAf6kf6sX6in+pn+rl+oV/qV/q1fqPf6nf6vf6gP+pP+rP+or/qb/q7/qF/6l86Qv/WASbQBJlgE2J8JtT4DTLYgCGGGma4EUYaZbQxxhpnPBPNRDcxTEwTy8Q2cUxcE8/ENwlMQpPIJDZJTFKTzCQ3KUxKk8qkNmlMWpPOpDcZTEaTyWQ2WUyYyWqymewmh8lpcpk8gblNHpPX5DP5TQFT0BQyhU0RU9QUM8VNCVPSlDKlTRlT1pQz5U0FU9FUMpVNFVPVVDPVTQ1TK+QvNc3aoD/9P/LX0X/tI2tUzNqgP7WuNc2/rzoT+Od6rZDTgf8Y/zU33987/LVqbdBfs/71TOQOprapY+qaeibcnAw5GVLfNDANTSPT2DQxTcz6oKammWluWpiWppVpbdqYtqadaW86mA6mo+lkOpsupqvparqZ7qaH6Wl+hfYyvU0fExHa1/Qz/c0A8zt0oBlkBpshZqgJ8A8zw80IM9KMMqPNGBPoH2vGmfFmgploJpnJZoqZaqaZ6WaGSe6faWaZ2WaOmWvmmflmgVloFpnFZolZapaZ5WaFWWlWmdVmjVlr1pn1ZoPZaDaZzWaL2Wq2me1mu9lhdpidZqfZZXabPSYsMCxwr9ln9psD5qA5ZA6bI+aoOWaOmxPmpDllTpsz5qw5Z86bC+aiuWQumyvmqrlmrpsb5qa5ZW6bO6ZIUJGgu+aeuW/umwfmgXloHpnH5ol5ap6Z5+aFeWFemqJBf7wyr80b89a8M+/NB/PRfDKfzRfz1Xwz38x38+NvP80vE2EiTLbAbIG/TcCf35zYIBtsQ6zPhlq/RRZbsMRSyyy3woYHSKustsZa66xno9noNobtEhLTxrSxbCJ/bBvHxrFxbTwbz8a3CWxCm9AmsoltEpvUJrPJbQqb0qayf+7i4QGpbRqb1qaz6W0Gm8FmtJlsZpvFhtmsNpvNbnPYnDaXzW3z2Dw2r81n89n8Nr8tYAvaQrawLWKL2mK2uC1hS9pStrQtY8vacra8rWAr2kq2sq1iq9pqtrqtYWvaWra2rWPr2no23Na3DWxD28g2tk1sU9vMNrctbEvbyra2bWxb2862tx1sR9vJdrZdbFfbzXa3PWxPm8Xfy/ayvW0f29f2s/3tADvQDrSD7GB7JnCIHWqz+YfZ4Ta7/y8j7Eg7yo62Y+xYO86OtxPsRDvJTrZT7FQ7zU63M+xMO8vOtnPsXDvPzrcL7EK7yC62S+xSW8C/zC63K+xKu8qutmvsWrvODg9ZbzfYjXaT3Wy32K12m91ud9iddpfdbffYvXaf3W8P2BEhB+0he9gesUftMXvcnrAn7Sl72p62Z+xZe86etxdsg8CL9pK9bK/Yq/aavW5v2Jv2lr1t79i79p69b/cGPbAP7SP72D6xT+0/6ree2ef2hX1pX9nX9o19a9/Z9/aD/Wg/2c/2i/1qv9nv9of9aX/ZCPvbBrgYAYHuTxVXkAv+W4jzuVDnd3WCG/v7ByD3RxP/H9h9DPgYAG5GCHHUMcedcNIpp51x1jnnuWguuovhYrpYLraL4+K6eC6+Cw9O4BJGSuQSuyQuqWvnT+aSuxQuPCClS+VSuzQujUvr0rn0LoPL4DK6TC6Ty+wyuywuzGV12Vx2l8PldLlcbpfH5XX5XH5XwBV0hVxhV8QVdcVccVfClXSlXGlXxpV15Vx5V8FVdJVcZVfF/akIaxFc1VVz1V0NV9PVcrVdHVfX1XPhrr5r4Bq6Rq6xa+KaumauuWvhWrpWrrVr49q6dq696+A6uk6us+viurpurrvr4Xq6Xq636+P6un6uvxvgBrpBbrAb4oa6YW64G+FGulFutBvjxrpxbryb4Ca6SW6ym+KmumluupvhZrpZbrab4+a6eW6+W+AWukVusVvilrplbrlb4Va6VW61W+PWunVuvdvgNrpNbrPb4ra6bW672+F2ul1ut9vj9rp9br874A66Q+6wO+KOumPuuDvhTrpT7rQ74866c+68u+Auukvusrvirrpr7rq74W66W+62u+PuunvuvnvgHrpH7rF74p66Z+65e+FeulfutXvj3rp37r374D66T+6z++K+um/uu/vhfrpfLsL9dgFeoBfkBXshns8L9fwe8rAHHvGoxzzuCU96ytOe8aznPM+L5kX3YngxvVhebC+OF9eL58X3EngJvUReYi+Jl9RL5iX3UngpvVReai+Nl9ZL56X3MngZvUxeZi+LF+Zl9bJ52b0cXk4vl5fby+Pl9fJ5+b0CXkGvkFfYK+IV9Yp5xb0SXkmvlFfaK+OV9cp55b0KXkWvklfZq+JV9ap51b0aXk2vllfbq+PV9ep54V59r4HX0GvkNfaaeE29Zl5zr4XX0mvltfbaeG29dl577/f/2jp4HbyOXievs9flnzP/A9yS5s/HXAMA"
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
var LIBLLAMA_VERSION = "b12077-b704a8b";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
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
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
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
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true
    });
    __publicField(this, "slotFileId", 0);
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
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
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
